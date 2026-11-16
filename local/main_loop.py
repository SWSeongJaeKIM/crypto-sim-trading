"""
PC에 상주시켜 계속 돌리는 메인 루프.

Claude/인터넷상의 AI를 전혀 호출하지 않는다. 이 파일 하나를
`python3 main_loop.py` 로 실행해두면(또는 OS 서비스로 등록) 다음을 반복한다:

  - 매 1분: 6코인 최신 1분봉 수집 -> 5개 전략(조건식 4 + LSTM 1)이 각자
    신호 계산 -> 모의 매수/매도 -> SQLite에 상태 저장
  - 매일 1회(기본 새벽 4시): 최근 N일 1분봉으로 LSTM 재학습

전략별로 포트폴리오를 2개씩(리셋용 + 누적용) 동시에 운영한다:
  - "<strategy>__reset"      : RESET_DAYS일마다 1000만원으로 초기화.
    "최근 N일 동안 이 전략이 잘하고 있나"를 보기 위한 단기 비교용.
    과거의 우연한 한 번의 거래가 전체 성과를 계속 왜곡하지 않도록 함.
  - "<strategy>__cumulative" : 처음 시작 이후 한 번도 리셋 안 함.
    "이 전략이 장기적으로 얼마나 벌었나"를 보기 위한 전체 기간용.
신호 계산(어떤 코인을 사고 팔지)은 두 포트폴리오가 동일하게 따르고,
각자의 가상 계좌 상태(현금/보유량)만 독립적으로 갖는다.

중지하려면 Ctrl+C. 로그는 콘솔과 local/run.log에 같이 남는다.
"""

from __future__ import annotations

import logging
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

import data_fetcher
import lstm_model
import storage
from portfolio import Portfolio, COINS, INITIAL_CAPITAL_KRW
from strategies import ALL_STRATEGIES

MIN_TRADE_FRACTION = 0.1      # 확신도가 낮을 때(0.0) 베팅 비율 — 그래도 조금은 움직임
MAX_TRADE_FRACTION = 0.9      # 확신도가 높을 때(1.0) 베팅 비율
CYCLE_INTERVAL_SEC = 60       # 매매 사이클 주기 (1분봉이므로 60초)
RETRAIN_TIMES_LOCAL = [
    (0, 0), (2, 24), (4, 48), (7, 12), (9, 36),
    (12, 0), (14, 24), (16, 48), (19, 12), (21, 36),
]  # 매일 이 (시,분) 10번(로컬 시간, 2시간24분 간격 균등분할) LSTM 재학습 — GPU 사용으로 재학습 비용이 줄어 더 자주 돌림
HISTORY_DAYS_FOR_TRAIN = 90   # LSTM 학습에 쓸 최근 일수 — GPU 사용으로 더 많은 데이터 처리 가능해져 30일->90일로 확대
STRATEGY_NAMES = list(ALL_STRATEGIES.keys()) + ["lstm_predictor"]
RESET_DAYS = 7                # "리셋" 포트폴리오를 며칠마다 1000만원으로 초기화할지
PORTFOLIO_MODES = ["reset", "cumulative"]
# 클라우드 6에이전트 분석팀(analysis_team/)이 GitHub Actions 스케줄로 매시간
# 이 분(UTC)에 도는데, 이 PC 루프는 그 스케줄을 실행하지 않고 "다음 실행이
# 몇 분 남았는지"만 콘솔에 보여주기 위해 참고한다(실제 실행 주기를 바꾸고
# 싶으면 클라우드 스케줄 작업 설정에서 바꾸고 여기 숫자도 맞춰줘야 함).
CLOUD_ANALYSIS_MINUTE_UTC = 17

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        # heartbeat.txt처럼 절대경로로 고정 (analysis_team execution 리포트
        # 2026-11-16_1422 제안). 상대경로("run.log")였을 때는 watchdog.bat나
        # OS 서비스가 local/이 아닌 다른 작업 디렉토리에서 프로세스를 띄우면
        # 모듈 docstring이 말하는 "local/run.log"와 다른 위치에 로그가 생겨
        # 문제 진단 시 로그를 못 찾는 문제가 있었다.
        logging.FileHandler(Path(__file__).parent / "run.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("main_loop")

_stop = False


def _handle_sigint(signum, frame):
    global _stop
    log.info("종료 신호 수신 — 이번 사이클 끝나면 정지합니다.")
    _stop = True


signal.signal(signal.SIGINT, _handle_sigint)
signal.signal(signal.SIGTERM, _handle_sigint)


HEARTBEAT_PATH = Path(__file__).parent / "heartbeat.txt"


def _touch_heartbeat() -> None:
    """watchdog.bat가 "멈췄는지"를 판단하는 유일한 근거. SQLite를 매번 열어
    파싱하는 대신, 파일 하나의 수정시각(mtime)만 보면 되게 가장 가벼운
    방식으로 분리했다. set_progress()가 상태 전환마다, 그리고 데이터 수집처럼
    오래 걸리는 단계 중간(fetch_all_minute_history의 on_progress 콜백)에도
    호출돼서, "원래 오래 걸리는 중"과 "진짜로 멈춤"을 구분할 수 있게 한다."""
    try:
        HEARTBEAT_PATH.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
    except OSError:
        pass  # 하트비트 파일 쓰기 실패는 워치독만 못 믿게 할 뿐, 매매 자체는 막지 않는다


def set_progress(stage: str, detail: str = "", **extra) -> None:
    """대시보드가 보여줄 '지금 뭐 하고 있는지' 상태. 매 단계 전환마다 갱신.
    stage: idle | collecting | training | cycle_running | sleeping | error

    주의: 커넥션을 파라미터로 받지 않고 매번 자체적으로 짧게 연다. 데이터
    수집/학습처럼 몇 분씩 걸리는 작업 중에도 이 함수가 자주 호출되는데,
    만약 바깥에서 전달받은 긴 커넥션에 썼다면 그 작업 전체 시간 동안 DB
    쓰기 잠금을 들고 있게 되어 동시에 도는 다른 프로세스(main_loop 사이클,
    dashboard, retrain_now.py 등)가 "database is locked"를 만난다. 매번
    새로 열고 즉시 commit+close하면 그 순간만 짧게 잠겼다가 바로 풀린다."""
    _touch_heartbeat()
    with storage.connect() as conn:
        storage.set_meta(conn, "progress", {
            "stage": stage,
            "detail": detail,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **extra,
        })


def portfolio_key(strategy_name: str, mode: str) -> str:
    """storage의 portfolios 테이블은 strategy 하나를 PK로 쓰므로,
    '<strategy>__<mode>' 형태로 이름 공간을 나눠 reset/cumulative를 구분한다."""
    return f"{strategy_name}__{mode}"


def load_or_new_portfolio(conn, strategy_name: str, mode: str = "cumulative") -> Portfolio:
    key = portfolio_key(strategy_name, mode)
    doc = storage.get_portfolio(conn, key)
    if doc:
        pf = Portfolio.from_doc(doc)
        if mode == "reset" and _reset_is_due(pf.created_at):
            log.info(f"[{key}] {RESET_DAYS}일 경과 — 포트폴리오 초기화")
            return Portfolio.new(key, coins=COINS, initial_capital_krw=INITIAL_CAPITAL_KRW)
        return pf
    return Portfolio.new(key, coins=COINS, initial_capital_krw=INITIAL_CAPITAL_KRW)


def _reset_is_due(created_at_iso: str) -> bool:
    try:
        created_at = datetime.fromisoformat(created_at_iso)
    except (TypeError, ValueError):
        return False
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - created_at >= timedelta(days=RESET_DAYS)


def _fmt_remaining(delta: timedelta) -> str:
    total_sec = max(0, int(delta.total_seconds()))
    h, rem = divmod(total_sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}시간 {m}분"
    if m:
        return f"{m}분 {s}초"
    return f"{s}초"


def _retrain_slots_today(now_local: datetime) -> list[datetime]:
    """오늘 날짜 기준 RETRAIN_TIMES_LOCAL 각 (시,분)을 datetime으로 변환한 목록.
    RETRAIN_TIMES_LOCAL은 시간순으로 정렬돼 있다고 가정(기본: 2시간24분 간격 10회)."""
    return [
        now_local.replace(hour=h, minute=m, second=0, microsecond=0)
        for h, m in RETRAIN_TIMES_LOCAL
    ]


def _last_due_slot(now_local: datetime) -> datetime:
    """지금(now_local) 기준으로 "가장 최근에 지나간 재학습 슬롯" 시각을 반환.
    오늘의 모든 슬롯이 아직 안 지났으면(예: 새벽 3시) 어제의 마지막 슬롯."""
    todays = _retrain_slots_today(now_local)
    passed = [t for t in todays if t <= now_local]
    if passed:
        return passed[-1]
    yesterday = now_local - timedelta(days=1)
    return _retrain_slots_today(yesterday)[-1]


def retrain_is_due(last_retrain_at) -> bool:
    """지금 LSTM 재학습을 돌려야 하는지 판단.

    하루 10번(기본 2시간24분 간격) 각 슬롯에 대해 "그 슬롯 시각이 이미 지났는데
    아직 그 슬롯만큼의 재학습을 안 했다"로 판단한다. "지금이 정확히 그
    시각이다"로 판단하면 PC가 꺼져 있거나 재시작 등으로 그 1분을 놓치는
    순간 다음 슬롯까지 영영 재학습을 못 하게 되는 문제가 있었다.
    last_retrain_at: 마지막으로 재학습을 "완료"한 시각(datetime, naive 로컬
    시간) 또는 None(한 번도 안 했음)."""
    now_local = datetime.now()
    due_slot = _last_due_slot(now_local)
    if last_retrain_at is None:
        return True
    return last_retrain_at < due_slot


def time_until_next_retrain(last_retrain_at) -> timedelta:
    """다음 LSTM 재학습까지 남은 시간(표시용).
    아직 안 한 지나간 슬롯이 있으면 "0초"(지금 바로) — 실제로 다음 사이클에
    바로 돌아간다. 이미 가장 최근 슬롯까지 했으면 그 다음 슬롯(오늘 나머지
    또는 내일 첫 슬롯)까지."""
    now_local = datetime.now()
    if retrain_is_due(last_retrain_at):
        return timedelta(seconds=0)
    upcoming = [t for t in _retrain_slots_today(now_local) if t > now_local]
    if upcoming:
        target = upcoming[0]
    else:
        target = _retrain_slots_today(now_local + timedelta(days=1))[0]
    return target - now_local


def last_retrain_at_from_storage(conn) -> datetime | None:
    """storage에 저장된 last_lstm_train 메타(UTC timestamp)를 읽어서,
    naive 로컬 datetime으로 변환해 반환한다. main_loop.py가 재시작돼도
    "마지막 재학습을 언제 했는지"를 메모리가 아니라 여기서 복원한다.
    기록이 없으면 None."""
    meta = storage.get_meta(conn, "last_lstm_train")
    if not meta or not meta.get("timestamp"):
        return None
    try:
        ts = datetime.fromisoformat(meta["timestamp"])
    except (TypeError, ValueError):
        return None
    if ts.tzinfo is not None:
        ts = ts.astimezone().replace(tzinfo=None)  # UTC -> 로컬 타임존(naive)
    return ts


def confidence_to_trade_fraction(confidence: float) -> float:
    """확신도(0.0~1.0)를 실제 베팅 비율(MIN_TRADE_FRACTION~MAX_TRADE_FRACTION)로
    변환. 확신도가 낮아도 0으로 만들지 않는 이유: 신호 자체(BUY/SELL)는 이미
    "조건을 만족했다"는 판단이므로 최소한의 베팅은 하고, 확신도는 그 위에
    강도를 더하는 역할만 한다."""
    confidence = max(0.0, min(1.0, confidence))
    return MIN_TRADE_FRACTION + (MAX_TRADE_FRACTION - MIN_TRADE_FRACTION) * confidence


def time_until_next_cloud_analysis() -> timedelta:
    """클라우드 6에이전트 분석팀의 다음 실행(매시간 CLOUD_ANALYSIS_MINUTE_UTC분,
    UTC 기준)까지 남은 시간. 실제 실행은 클라우드에서 일어나고, 이 함수는
    표시용으로 그 스케줄을 로컬에서 계산만 한다."""
    now_utc = datetime.now(timezone.utc)
    target = now_utc.replace(minute=CLOUD_ANALYSIS_MINUTE_UTC, second=0, microsecond=0)
    if target <= now_utc:
        target += timedelta(hours=1)
    return target - now_utc


def run_one_cycle(conn) -> None:
    now = datetime.now(timezone.utc).isoformat()
    set_progress("collecting", "최신 1분봉 수집 중...")
    candles = data_fetcher.fetch_all_latest_candles(count=max(lstm_model.SEQ_LEN + 5, 200))
    if not candles:
        log.warning("캔들 수집 실패 — 이번 사이클 스킵")
        set_progress("error", "캔들 수집 실패, 다음 사이클에 재시도")
        return

    set_progress("cycle_running", "5개 전략 신호 계산/매매 중...")

    prices_krw = {coin: float(df["close"].iloc[-1]) for coin, df in candles.items() if len(df)}

    # 지난 사이클에 LSTM이 냈던 예측들을 이번 사이클의 실제가로 확정 (적중여부 계산)
    n_resolved = storage.resolve_predictions(conn, prices_krw)
    if n_resolved:
        log.info(f"LSTM 예측 {n_resolved}건 확정")

    for strat_name in STRATEGY_NAMES:
        # 전략 하나당 try/except로 격리한다. run_one_cycle() 전체가
        # storage.connect()의 with 블록(한 트랜잭션) 안에서 도는데, 예전에는
        # 이 루프 안에서 예외가 하나라도 나면(예: 모델 파일 손상, 전략 내부
        # 버그) 그 예외가 run_one_cycle()까지 그대로 전파되어 storage.connect()가
        # rollback()을 호출했다 — 즉 이미 정상적으로 처리되어
        # set_portfolio/add_trades/add_snapshot까지 호출된 "먼저 도는
        # 전략들"의 이번 사이클 매매 기록까지 통째로 날아갔다. 전략별로
        # 감싸서, 하나가 실패해도 이미 처리된 다른 전략들의 commit은 그대로
        # 살리고 실패한 전략만 이번 사이클 건너뛴다.
        try:
            if strat_name == "lstm_predictor":
                strat = lstm_model.LSTMStrategy(coin=None)  # coin은 코인별로 아래서 바꿔 끼움
            else:
                params = storage.get_params(conn, strat_name) or ALL_STRATEGIES[strat_name].default_params()
                strat = ALL_STRATEGIES[strat_name](**params)

            # 신호(BUY/SELL/HOLD)와 확신도는 전략당 1번만 계산하고, reset/cumulative
            # 두 포트폴리오에 동일하게 적용한다 (매매 판단 로직은 같고, 계좌
            # 상태만 분리). 확신도(0~1)는 베팅 비율(MIN~MAX_TRADE_FRACTION)로
            # 변환되어 "얼마나 베팅할지"를 결정한다 — 애매한 신호는 조금만,
            # 뚜렷한 신호는 많이 움직인다.
            portfolios = {mode: load_or_new_portfolio(conn, strat_name, mode) for mode in PORTFOLIO_MODES}
            signals: dict[str, tuple[str, float]] = {}  # coin -> (signal, confidence)

            for coin, df in candles.items():
                if coin not in prices_krw or df.empty:
                    continue
                if strat_name == "lstm_predictor":
                    strat.coin = coin
                if len(df) < strat.min_bars():
                    continue
                signal_, confidence = strat.signal_with_confidence(df)
                signals[coin] = (signal_, confidence)
                price = prices_krw[coin]

                if strat_name == "lstm_predictor":
                    prob_up = lstm_model.predict_proba(coin, df)
                    if prob_up is not None:
                        storage.add_prediction(conn, coin, now, prob_up, price)

            for mode, pf in portfolios.items():
                # 1) 손절/강제청산 체크를 신호 적용보다 먼저 한다 — 이번 사이클의
                # 신호가 뭐든 상관없이, 지금 들고 있는 포지션이 이미 손절선을
                # 넘었으면 먼저 정리한다 (예: SELL 신호가 또 나와도 숏을 늘리지
                # 않고, 이미 위험 수준인 포지션부터 끊는다).
                for coin, price in prices_krw.items():
                    if coin not in pf.slots:
                        continue
                    stop_reason = pf.check_stop_loss_and_liquidation(coin, price)
                    if stop_reason:
                        log.warning(f"[{portfolio_key(strat_name, mode)}] {stop_reason}")

                # 2) 신호 적용 (롱/숏 전환은 apply_buy_signal/apply_sell_signal이
                # 처리 — BUY는 숏 정리 후 롱, SELL은 롱 정리 후 숏).
                for coin, (signal_, confidence) in signals.items():
                    if coin not in pf.slots:
                        continue
                    price = prices_krw[coin]
                    fraction = confidence_to_trade_fraction(confidence)
                    if signal_ == "BUY":
                        pf.apply_buy_signal(coin, price, fraction,
                                             reason=f"{strat_name} BUY (확신도 {confidence:.2f} -> {fraction*100:.0f}%)")
                    elif signal_ == "SELL":
                        pf.apply_sell_signal(coin, price, fraction,
                                              reason=f"{strat_name} SELL (확신도 {confidence:.2f} -> {fraction*100:.0f}%)")

                key = portfolio_key(strat_name, mode)
                storage.set_portfolio(conn, key, pf.to_doc(), now)
                storage.add_trades(conn, key, pf.new_trades_as_dicts())
                total_value = pf.total_value_krw(prices_krw)
                pnl_pct = pf.pnl_pct(prices_krw)
                storage.add_snapshot(conn, key, now, total_value, pnl_pct)
        except Exception as e:
            log.exception(f"[{strat_name}] 전략 처리 중 에러, 이번 사이클은 건너뜀 "
                           f"(이미 처리된 다른 전략들의 매매 기록에는 영향 없음): {e}")
            continue

    storage.set_meta(conn, "status", {"last_run_at": now, "prices_krw": prices_krw, "source": "upbit"})
    log.info(f"사이클 완료 @ {now} | 가격 샘플 BTC={prices_krw.get('BTC')}")


def retrain_lstm() -> None:
    """LSTM 재학습 1회 실행.

    주의: 커넥션을 파라미터로 받지 않는다. 데이터 수집(네트워크 I/O, 수십초~분)과
    학습(8 epoch, 수분)은 DB를 전혀 건드리지 않고 돌고, 결과를 실제로 저장해야
    하는 짧은 순간에만 그때그때 storage.connect()를 연다. 예전에는 이 함수
    전체가 하나의 커넥션을 받아서 처음부터 끝까지 들고 있었는데, 그러면 이
    함수가 도는 몇 분 내내 DB 쓰기 잠금이 풀리지 않아 main_loop 사이클이나
    retrain_now.py를 동시에 돌리면 "database is locked"가 났다."""
    log.info(f"LSTM 재학습 시작 (최근 {HISTORY_DAYS_FOR_TRAIN}일 1분봉)...")
    results = []

    def _on_fetch_progress(coin, i, total):
        set_progress("training", f"{coin} 데이터 수집 중... ({i}/{total})",
                     current_coin=coin, coin_index=i, coin_total=total, train_step="fetching")

    history = data_fetcher.fetch_all_minute_history(days=HISTORY_DAYS_FOR_TRAIN, on_progress=_on_fetch_progress)

    for i, (coin, df) in enumerate(history.items(), start=1):
        set_progress("training", f"{coin} 모델 학습 중... ({i}/{len(history)})",
                     current_coin=coin, coin_index=i, coin_total=len(history), train_step="fitting")

        def _on_epoch(_coin, epoch_idx, total_epochs, _i=i):
            set_progress("training", f"{_coin} 모델 학습 중... epoch {epoch_idx}/{total_epochs}",
                         current_coin=_coin, coin_index=_i, coin_total=len(history), train_step="fitting")

        r = lstm_model.train_model(coin, df, epochs=8, history_days=HISTORY_DAYS_FOR_TRAIN, on_epoch=_on_epoch)
        results.append(r)
        log.info(f"  {r}")
        # 학습할 때마다 누적 기록 (덮어쓰지 않음) — 대시보드에서 정확도 추이를 보기 위함
        # 이 코인 학습이 막 끝난 짧은 순간에만 connect해서 바로 commit+close.
        with storage.connect() as conn:
            storage.add_model_training(conn, {
                "coin": coin,
                "timestamp": r.get("timestamp", datetime.now(timezone.utc).isoformat()),
                "model_type": "lstm_finetune" if r.get("finetuned") else "lstm_initial",
                "feature_cols": lstm_model.FEATURE_COLS,
                "seq_len": r.get("seq_len", lstm_model.SEQ_LEN),
                "history_days": HISTORY_DAYS_FOR_TRAIN,
                "data_start": r.get("data_start"),
                "data_end": r.get("data_end"),
                "n_samples": r.get("n_samples"),
                "n_train": r.get("n_train"),
                "n_val": r.get("n_val"),
                "val_accuracy": r.get("val_accuracy"),
                "epochs": r.get("epochs"),
                "trained": r.get("trained", False),
                "reason": r.get("reason", ""),
            })

    with storage.connect() as conn:
        storage.set_meta(conn, "last_lstm_train", {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "results": results,
        })
    set_progress("idle", "학습 완료, 대기 중")


def main():
    log.info("=== crypto-sim-trading 로컬 루프 시작 ===")
    log.info(f"대상 코인: {COINS}")
    log.info(f"전략: {STRATEGY_NAMES}")

    # 중요: 커넥션을 while 루프 전체에서 재사용하지 않는다. storage.connect()는
    # with 블록을 빠져나갈 때(= 루프 전체가 끝날 때)만 commit하는 context
    # manager라서, 여기서 한 번만 열면 프로그램이 종료될 때까지 하나의 쓰기
    # 트랜잭션이 영원히 안 끝난 채로 유지된다 — 그동안 dashboard.py 등 다른
    # 프로세스가 같은 trading.db에 쓰려고 하면 "database is locked"가 난다.
    # 그래서 사이클/재학습 등 "한 단위의 작업"마다 매번 새로 connect해서
    # 그 단위가 끝나면 즉시 commit+close 하도록 바꿨다.

    # 최초 1회: LSTM 모델이 하나도 없으면 바로 학습부터.
    # while 루프 안의 재학습은 try/except로 보호되지만 이 최초 학습은
    # 보호가 없어서, 데이터 수집 실패 등으로 예외가 나면 프로그램 전체가
    # 죽는 문제가 있었다 — 모델 없이도 일단 매매 루프(신호는 HOLD로만
    # 나가지만 조건식 전략은 정상 동작)는 돌 수 있게 여기서도 막는다.
    has_any_model = any((lstm_model.MODEL_DIR / f"{c}_lstm.pt").exists() for c in COINS)
    if not has_any_model:
        log.info("저장된 LSTM 모델이 없어 최초 학습을 먼저 수행합니다 (몇 분 걸릴 수 있음)...")
        try:
            retrain_lstm()  # 이 안에서 last_lstm_train 메타를 저장함 (내부에서 짧게 connect)
        except Exception as e:
            log.exception(f"최초 LSTM 학습 중 에러 (LSTM 없이 조건식 전략만으로 계속 진행): {e}")

    # 메모리 변수가 아니라 storage에 저장된 마지막 재학습 시각으로
    # 시작한다 — main_loop.py를 재시작해도 "이번 슬롯을 이미 했는지"를 안다.
    # (바로 위에서 최초 학습을 했다면 그 기록이 여기서 바로 "방금"으로 잡힌다.)
    with storage.connect() as conn:
        last_retrain_at = last_retrain_at_from_storage(conn)
    if last_retrain_at:
        log.info(f"마지막 LSTM 재학습: {last_retrain_at} (저장된 기록 기준)")
    log.info(f"LSTM 재학습 스케줄: 매일 {', '.join(f'{h:02d}:{m:02d}' for h, m in RETRAIN_TIMES_LOCAL)} (하루 {len(RETRAIN_TIMES_LOCAL)}회)")

    while not _stop:
        try:
            with storage.connect() as conn:
                run_one_cycle(conn)
        except Exception as e:
            log.exception(f"사이클 실행 중 에러 (스킵하고 계속): {e}")

        now_local = datetime.now()
        if retrain_is_due(last_retrain_at):
            try:
                retrain_lstm()  # 내부에서 단계별로 짧게 connect — 재학습 도는 동안 DB를 안 잡음
                last_retrain_at = now_local
            except Exception as e:
                log.exception(f"LSTM 재학습 중 에러: {e}")

        retrain_left = _fmt_remaining(time_until_next_retrain(last_retrain_at))
        analysis_left = _fmt_remaining(time_until_next_cloud_analysis())
        log.info(f"다음 LSTM 재학습까지 {retrain_left} | 다음 클라우드 분석팀 실행까지 {analysis_left}")

        for _ in range(CYCLE_INTERVAL_SEC):
            if _stop:
                break
            time.sleep(1)

    log.info("=== 루프 정상 종료 ===")


if __name__ == "__main__":
    main()
