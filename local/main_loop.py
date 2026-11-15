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

import pandas as pd

import data_fetcher
import lstm_model
import storage
from portfolio import Portfolio, COINS, INITIAL_CAPITAL_KRW
from strategies import ALL_STRATEGIES

MIN_TRADE_FRACTION = 0.2      # 확신도가 낮을 때(0.0) 베팅 비율 — 그래도 조금은 움직임
MAX_TRADE_FRACTION = 0.9      # 확신도가 높을 때(1.0) 베팅 비율
CYCLE_INTERVAL_SEC = 60       # 매매 사이클 주기 (1분봉이므로 60초)
RETRAIN_HOUR_LOCAL = 4        # 매일 이 시(0~23, 로컬 시간)에 LSTM 재학습
HISTORY_DAYS_FOR_TRAIN = 30   # LSTM 학습에 쓸 최근 일수
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
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("run.log", encoding="utf-8")],
)
log = logging.getLogger("main_loop")

_stop = False


def _handle_sigint(signum, frame):
    global _stop
    log.info("종료 신호 수신 — 이번 사이클 끝나면 정지합니다.")
    _stop = True


signal.signal(signal.SIGINT, _handle_sigint)
signal.signal(signal.SIGTERM, _handle_sigint)


def set_progress(conn, stage: str, detail: str = "", **extra) -> None:
    """대시보드가 보여줄 '지금 뭐 하고 있는지' 상태. 매 단계 전환마다 갱신.
    stage: idle | collecting | training | cycle_running | sleeping | error"""
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


def time_until_next_retrain(last_retrain_date) -> timedelta:
    """다음 LSTM 재학습(매일 RETRAIN_HOUR_LOCAL시, 로컬시간)까지 남은 시간.
    last_retrain_date가 오늘이면 다음 재학습은 내일 이 시각."""
    now_local = datetime.now()
    target = now_local.replace(hour=RETRAIN_HOUR_LOCAL, minute=0, second=0, microsecond=0)
    if target <= now_local or last_retrain_date == now_local.date():
        target += timedelta(days=1)
    return target - now_local


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
    set_progress(conn, "collecting", "최신 1분봉 수집 중...")
    candles = data_fetcher.fetch_all_latest_candles(count=max(lstm_model.SEQ_LEN + 5, 200))
    if not candles:
        log.warning("캔들 수집 실패 — 이번 사이클 스킵")
        set_progress(conn, "error", "캔들 수집 실패, 다음 사이클에 재시도")
        return

    set_progress(conn, "cycle_running", "5개 전략 신호 계산/매매 중...")

    prices_krw = {coin: float(df["close"].iloc[-1]) for coin, df in candles.items() if len(df)}

    # 지난 사이클에 LSTM이 냈던 예측들을 이번 사이클의 실제가로 확정 (적중여부 계산)
    n_resolved = storage.resolve_predictions(conn, prices_krw)
    if n_resolved:
        log.info(f"LSTM 예측 {n_resolved}건 확정")

    for strat_name in STRATEGY_NAMES:
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
            for coin, (signal_, confidence) in signals.items():
                if coin not in pf.slots:
                    continue
                price = prices_krw[coin]
                fraction = confidence_to_trade_fraction(confidence)
                if signal_ == "BUY":
                    pf.buy(coin, price, pct_of_cash=fraction,
                           reason=f"{strat_name} BUY (확신도 {confidence:.2f} -> {fraction*100:.0f}%)")
                elif signal_ == "SELL":
                    pf.sell(coin, price, pct_of_qty=fraction,
                            reason=f"{strat_name} SELL (확신도 {confidence:.2f} -> {fraction*100:.0f}%)")

            key = portfolio_key(strat_name, mode)
            storage.set_portfolio(conn, key, pf.to_doc(), now)
            storage.add_trades(conn, key, pf.new_trades_as_dicts())
            total_value = pf.total_value_krw(prices_krw)
            pnl_pct = pf.pnl_pct(prices_krw)
            storage.add_snapshot(conn, key, now, total_value, pnl_pct)

    storage.set_meta(conn, "status", {"last_run_at": now, "prices_krw": prices_krw, "source": "upbit"})
    log.info(f"사이클 완료 @ {now} | 가격 샘플 BTC={prices_krw.get('BTC')}")


def retrain_lstm(conn) -> None:
    log.info(f"LSTM 재학습 시작 (최근 {HISTORY_DAYS_FOR_TRAIN}일 1분봉)...")
    results = []

    def _on_fetch_progress(coin, i, total):
        set_progress(conn, "training", f"{coin} 데이터 수집 중... ({i}/{total})",
                     current_coin=coin, coin_index=i, coin_total=total, train_step="fetching")

    history = data_fetcher.fetch_all_minute_history(days=HISTORY_DAYS_FOR_TRAIN, on_progress=_on_fetch_progress)

    for i, (coin, df) in enumerate(history.items(), start=1):
        set_progress(conn, "training", f"{coin} 모델 학습 중... ({i}/{len(history)})",
                     current_coin=coin, coin_index=i, coin_total=len(history), train_step="fitting")
        r = lstm_model.train_model(coin, df, epochs=8, history_days=HISTORY_DAYS_FOR_TRAIN)
        results.append(r)
        log.info(f"  {r}")
        # 학습할 때마다 누적 기록 (덮어쓰지 않음) — 대시보드에서 정확도 추이를 보기 위함
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

    storage.set_meta(conn, "last_lstm_train", {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
    })
    set_progress(conn, "idle", "학습 완료, 대기 중")


def main():
    log.info("=== crypto-sim-trading 로컬 루프 시작 ===")
    log.info(f"대상 코인: {COINS}")
    log.info(f"전략: {STRATEGY_NAMES}")

    with storage.connect() as conn:
        # 최초 1회: LSTM 모델이 하나도 없으면 바로 학습부터
        has_any_model = any((lstm_model.MODEL_DIR / f"{c}_lstm.pt").exists() for c in COINS)
        if not has_any_model:
            log.info("저장된 LSTM 모델이 없어 최초 학습을 먼저 수행합니다 (몇 분 걸릴 수 있음)...")
            retrain_lstm(conn)

        last_retrain_date = None
        while not _stop:
            try:
                run_one_cycle(conn)
            except Exception as e:
                log.exception(f"사이클 실행 중 에러 (스킵하고 계속): {e}")

            now_local = datetime.now()
            if now_local.hour == RETRAIN_HOUR_LOCAL and last_retrain_date != now_local.date():
                try:
                    retrain_lstm(conn)
                    last_retrain_date = now_local.date()
                except Exception as e:
                    log.exception(f"LSTM 재학습 중 에러: {e}")

            retrain_left = _fmt_remaining(time_until_next_retrain(last_retrain_date))
            analysis_left = _fmt_remaining(time_until_next_cloud_analysis())
            log.info(f"다음 LSTM 재학습까지 {retrain_left} | 다음 클라우드 분석팀 실행까지 {analysis_left}")

            for _ in range(CYCLE_INTERVAL_SEC):
                if _stop:
                    break
                time.sleep(1)

    log.info("=== 루프 정상 종료 ===")


if __name__ == "__main__":
    main()
