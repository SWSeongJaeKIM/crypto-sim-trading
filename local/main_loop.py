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
import logging.handlers
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
PREDICTIONS_PRUNE_KEEP_DAYS = 90  # predictions 테이블에서 확정(resolved=1)되고 이보다 오래된 행을
                                   # 하루 1회 삭제 (analysis_team 리포트 2026-11-21_2118_storage.md,
                                   # synthesizer Top3 #3) — 안 그러면 사이클마다(1분 간격) 영구
                                   # 누적되어 DB 파일 크기가 무한정 커진다.
SNAPSHOTS_PRUNE_KEEP_DAYS = 365   # snapshots 테이블도 predictions와 동일한 append-only 누적
                                   # 패턴이라 보관주기를 둔다 (analysis_team 리포트
                                   # 2026-11-21_2221_storage.md). predictions(90일)보다 길게 잡아
                                   # 장기 pnl 추이를 직접 확인할 여지를 더 남겨둔다.
MODEL_TRAININGS_PRUNE_KEEP_DAYS = 365  # model_trainings도 동일한 append-only 누적
                                   # 패턴이라 보관주기를 둔다 (analysis_team 리포트
                                   # 2026-11-22_0221_storage.md, synthesizer Top3 #1). 성공한
                                   # 학습 기록(trained=1)은 모델 성능 추이 보존을 위해 절대
                                   # 지우지 않고, 실패 기록(trained=0)만 대상으로 한다 — 자세한
                                   # 이유는 storage.prune_model_trainings() docstring 참고.
STRATEGY_NAMES = list(ALL_STRATEGIES.keys()) + ["lstm_predictor"]
RESET_DAYS = 7                # "리셋" 포트폴리오를 며칠마다 1000만원으로 초기화할지
PORTFOLIO_MODES = ["reset", "cumulative"]
# 평상시(재학습이 섞이지 않은) 사이클 처리시간이 이 값(초)을 넘으면 경고 로그를
# 남긴다. 재학습 구간은 이미 610행 근처에서 소요시간을 별도로 로깅하므로 대상이
# 아니다. 이전에는 "_elapsed"(651행 근처)가 60초 주기 보정에만 쓰이고 로그로
# 남지 않아, 네트워크 지연 누적이나 전략 수 증가로 평상시 사이클이 점점 느려져
# 60초에 근접해가는 드리프트 조짐을 재학습 로그와 달리 사전에 포착할 방법이
# 없었다 (analysis_team execution 리포트 2026-11-22_0018 발견점 #2,
# synthesizer Top3 #1). 매 사이클(1분 간격) 무조건 찍으면 로그가 금방 폭주하므로
# 임계치 초과 시에만 남긴다.
CYCLE_ELAPSED_WARN_THRESHOLD_SEC = 30
# "영구 이어학습(finetune=True 기본값)" 구조에서는 전체 재학습(finetune=False)이
# 자동으로 트리거되지 않아, 연속 파인튠이 계속 누적되면 가중치가 특정 구간
# 패턴에 고착될 위험이 있다 — lstm_model.py가 이미 meta.json에
# finetune_chain_length(연속 파인튠 횟수)를 기록만 해두고 있었다(analysis_team
# lstm 리포트 2026-11-21_2118 발견점, 2026-11-22_0018 발견점 #3). 이 값이 아래
# 임계값 이상이면 retrain_lstm()이 그 코인만 finetune=False로 강제 전환해
# 처음부터 다시 학습한다(analysis_team lstm 리포트 2026-11-22_0018,
# synthesizer Top3 #2). 하루 최대 RETRAIN_TIMES_LOCAL(10)회 중 한 슬롯만큼의
# 추가 학습 비용이 코인별로 가끔 드는 것 외에는 기존 파인튠 로직을 바꾸지 않음.
FINETUNE_CHAIN_LENGTH_RESET_THRESHOLD = 30
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
        # 24/7 상주 루프라 회전(rotation) 없이는 run.log가 수개월~수년간
        # 끝없이 커진다 (analysis_team 리포트 2026-11-21_2221_execution.md).
        # 10MB마다 회전, 과거분 5개(최대 약 50MB)까지만 보존한다 — 기존
        # FileHandler와 같은 파일에 같은 포맷으로 계속 추가(append)되므로
        # 재시작 시 동작은 그대로이고, 다음 회전 시점부터만 분할된다.
        logging.handlers.RotatingFileHandler(
            Path(__file__).parent / "run.log", encoding="utf-8",
            maxBytes=10 * 1024 * 1024, backupCount=5,
        ),
    ],
)
log = logging.getLogger("main_loop")

_stop = False

# run_one_cycle()의 코인별 except(아래)에서 "모델 파일 없음"에 대해 최초
# 1회만 log.warning을 남기기 위한 메모리 플래그. has_any_model이 "코인 중
# 하나라도 모델이 있으면 True"라서 특정 코인만 모델이 손상/삭제돼도 최초
# 학습이 트리거되지 않던 문제(아래 missing_models 처리로 수정)와 별개로,
# 그 코인의 신호 계산은 다음 정기 재학습 슬롯(최대 2시간24분 후)까지 매
# 사이클(1분 간격) 계속 실패하므로, 그때마다 log.exception으로 전체
# 스택트레이스를 반복 적재하면 로그가 금방 폭주한다 (analysis_team
# execution 리포트 2026-11-16_1718 제안).
_warned_missing_model: set[str] = set()

# run_one_cycle()의 코인별 except(아래)에서 FileNotFoundError 외의 "반복되는"
# 예외(예: 모델 파일 손상으로 인한 RuntimeError, 특정 코인 데이터 형태 문제로
# 전략 내부에서 매 사이클 계속 터지는 예외)에 대해, 매 사이클(1분 간격)마다
# 전체 스택트레이스를 log.exception으로 반복 적재하지 않도록 한다. 키는
# (전략, 코인, 예외타입)이고 값은 마지막으로 전체 트레이스를 남긴 시각이다.
# FileNotFoundError와 달리 "조용히 끄기"가 아니라 "일정 시간마다 1회만 전체
# 트레이스, 그 사이엔 짧은 warning"으로 완화한다 — 실제로 지속되는 문제를
# 운영자가 너무 늦게 알아차리는 위험을 줄이기 위함이다 (analysis_team
# execution 리포트 2026-11-21_2320 제안).
_warned_repeated_exception: dict[tuple[str, str, str], float] = {}
_REPEATED_EXCEPTION_LOG_INTERVAL_SEC = 3600  # 1시간(60사이클)마다 전체 트레이스 1회

# run_one_cycle()에서 특정 코인만 이번 사이클 캔들 수집에 실패하면(예: Upbit
# API 일시 오류로 그 코인만 None), prices_krw/candles에서 조용히 빠지고
# 그 코인은 신호 계산뿐 아니라 손절/강제청산 체크(435행 근처)까지 이번
# 사이클엔 전혀 돌지 않는다. 지금까지는 이 상황이 로그에 전혀 남지 않아
# 사후 진단이 불가능했다 (analysis_team execution 리포트 2026-11-22_0417
# 발견점 #2, synthesizer Top3 #2). data_fetcher._log_fetch_exception과
# 동일한 방식(코인별 1시간 쓰로틀)으로 로그 폭주 없이 추적 가능하게 한다.
_warned_missing_coin_data: dict[str, float] = {}
_MISSING_COIN_DATA_LOG_INTERVAL_SEC = 3600


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
    storage.make_portfolio_key()로 '<strategy>__<mode>' 형태의 이름 공간을
    만들어 reset/cumulative를 구분한다 (조립 로직을 storage.py 한 곳으로
    모음 — analysis_team 리포트 2026-11-16_1619_storage.md)."""
    return storage.make_portfolio_key(strategy_name, mode)


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


def last_predictions_prune_date_from_storage(conn) -> str | None:
    """storage에 저장된 last_predictions_prune 메타에서 마지막으로 predictions
    보관주기 정리를 실행한 로컬 날짜(YYYY-MM-DD)를 읽어온다. 기록이 없으면 None
    (한 번도 실행 안 함 — 바로 오늘 실행하게 됨)."""
    meta = storage.get_meta(conn, "last_predictions_prune")
    if not meta:
        return None
    return meta.get("date")


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

    # 코인 중 일부만 이번 사이클 가격 데이터가 없으면(수집 실패 등), 그
    # 코인의 신호 계산과 손절체크가 모두 조용히 스킵된다 — 다음 사이클(최대
    # 1분 후)에 복구되지만, 1시간에 1회만 경고를 남겨 사후 진단이 가능하게
    # 한다 (analysis_team execution 리포트 2026-11-22_0417 발견점 #2,
    # synthesizer Top3 #2).
    missing_coins = sorted(set(COINS) - set(prices_krw.keys()))
    if missing_coins:
        now_mono = time.monotonic()
        for coin in missing_coins:
            last_logged = _warned_missing_coin_data.get(coin)
            if last_logged is None or (now_mono - last_logged) >= _MISSING_COIN_DATA_LOG_INTERVAL_SEC:
                log.warning(f"[{coin}] 이번 사이클 가격 데이터 없음 — 신호 계산 및 손절/강제청산 "
                             f"체크 모두 스킵됨 (다음 사이클에 복구 시도, 같은 코인은 "
                             f"{_MISSING_COIN_DATA_LOG_INTERVAL_SEC}초 내 반복 경고 생략)")
                _warned_missing_coin_data[coin] = now_mono
    else:
        _warned_missing_coin_data.clear()  # 전 코인 복구됐으니 다음 결손 시 즉시 다시 경고

    # 지난 사이클에 LSTM이 냈던 예측들을 이번 사이클의 실제가로 확정 (적중여부 계산)
    # try/except로 감싼다 (analysis_team execution 리포트 2026-11-21_2320 제안):
    # 이 호출이 전략 루프보다 "먼저" 실행되는데, 감싸지 않으면 여기서 예외가
    # storage.connect()의 with 블록까지 전파되어 rollback()이 호출되고, 아직
    # 시작도 안 한 전략 루프 자체가 통째로 스킵된다 — 전략별 격리(아래 305번째
    # 줄부터)를 만든 목적과 같은 이유로, 전략 루프 밖의 이 호출도 보호해야 한다.
    # 예측 확정은 다음 사이클(1분 후)에도 동일 가격 데이터로 재시도 가능한
    # 멱등적 동작이라, 이번 사이클에서 조용히 스킵돼도 매매 로직에는 영향이 없다.
    try:
        n_resolved = storage.resolve_predictions(conn, prices_krw)
        if n_resolved:
            log.info(f"LSTM 예측 {n_resolved}건 확정")
    except Exception as e:
        log.exception(f"LSTM 예측 확정 중 에러, 이번 사이클은 스킵 (다음 사이클에 재시도, "
                       f"전략 매매 로직에는 영향 없음): {e}")

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
                try:
                    # 코인 단위로도 격리한다 (analysis_team execution 리포트
                    # 2026-11-16_1422 제안). 위의 전략 단위 try/except만으로는,
                    # 예를 들어 ADA의 LSTM 모델 파일이 손상되어 predict_proba나
                    # signal_with_confidence에서 예외가 나면 이미 신호를 계산한
                    # BTC/ETH 등 나머지 코인들까지 이번 사이클에서 strat_name
                    # 전체가 건너뛰어졌다. 코인 하나의 장애가 같은 전략의 다른
                    # 코인들까지 막지 않도록, 신호 계산을 코인별로도 감싼다.
                    signal_, confidence = strat.signal_with_confidence(df)
                    signals[coin] = (signal_, confidence)
                    price = prices_krw[coin]

                    if strat_name == "lstm_predictor":
                        prob_up = lstm_model.predict_proba(coin, df)
                        if prob_up is not None:
                            storage.add_prediction(conn, coin, now, prob_up, price)
                except FileNotFoundError as e:
                    # 모델 파일 없음(코인별 결손)은 다음 재학습 슬롯까지 매
                    # 사이클 반복될 수 있는 예측 가능한 상태라, 코인별로 최초
                    # 1회만 짧게 warning을 남기고 이후엔 조용히 건너뛴다.
                    # 그 외 예외는 아래에서 그대로 log.exception을 유지한다.
                    if coin not in _warned_missing_model:
                        log.warning(f"[{strat_name}/{coin}] 모델 파일 없음, 다음 재학습 전까지 "
                                     f"이 코인만 계속 건너뜀 (같은 전략의 다른 코인에는 영향 "
                                     f"없음, 이후 동일 사이클 반복은 로그에 남기지 않음): {e}")
                        _warned_missing_model.add(coin)
                    continue
                except Exception as e:
                    # FileNotFoundError(위)와 달리 "조용히"가 아니라 "N초마다 1회만
                    # 전체 트레이스" — 예외 타입까지 키에 포함해, 같은 코인에서
                    # 다른 종류의 새 문제가 생기면 그건 바로 전체 트레이스로
                    # 남긴다(analysis_team execution 리포트 2026-11-21_2320 제안).
                    repeat_key = (strat_name, coin, type(e).__name__)
                    last_logged = _warned_repeated_exception.get(repeat_key)
                    now_mono = time.monotonic()
                    if last_logged is None or (now_mono - last_logged) >= _REPEATED_EXCEPTION_LOG_INTERVAL_SEC:
                        log.exception(f"[{strat_name}/{coin}] 신호 계산 중 에러, 이 코인만 이번 "
                                       f"사이클 건너뜀 (같은 전략의 다른 코인에는 영향 없음): {e}")
                        _warned_repeated_exception[repeat_key] = now_mono
                    else:
                        log.warning(f"[{strat_name}/{coin}] 신호 계산 중 에러 반복 중 (같은 종류 "
                                     f"{_REPEATED_EXCEPTION_LOG_INTERVAL_SEC}초 내 전체 트레이스는 "
                                     f"이미 남겼음, 이 코인만 이번 사이클 건너뜀): {type(e).__name__}: {e}")
                    continue

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

    # try/except로 감싼다 (analysis_team execution 리포트 2026-11-21_2320 제안):
    # 이 호출은 전략 루프 "뒤"에 있어, 전략별 try/except(위 315번째 줄부터)가
    # 이미 각 전략의 매매 기록 commit을 보존해준 뒤에도 여기서 예외가 나면
    # storage.connect()가 그 사이클 전체(이미 처리된 모든 전략의 매매 기록
    # 포함)를 rollback() 해버린다. status 메타는 대시보드 표시용이고 다음
    # 사이클에 다시 갱신되는 멱등적 동작이라, 실패해도 조용히 스킵하는 쪽이
    # 안전하다.
    try:
        storage.set_meta(conn, "status", {"last_run_at": now, "prices_krw": prices_krw, "source": "upbit"})
    except Exception as e:
        log.exception(f"상태 메타(status) 갱신 중 에러, 이번 사이클은 스킵 (다음 사이클에 재시도, "
                       f"이미 처리된 전략들의 매매 기록에는 영향 없음): {e}")
    log.info(f"사이클 완료 @ {now} | 가격 샘플 BTC={prices_krw.get('BTC')}")


def retrain_lstm() -> bool:
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

    # fetch 단계와 train 단계 소요시간을 각각 남긴다. 기존에는 호출부(main()의
    # retrain_lstm() 주변)에서 전체 묶음 소요시간만 로그로 남겨, "사이클이 느려진
    # 게 90일치 재수집 쪽인지 학습 자체인지" 구분할 수 없었다 (analysis_team
    # execution 리포트 2026-11-22_0417 구체적 제안 #3).
    _fetch_started = time.monotonic()
    history = data_fetcher.fetch_all_minute_history(days=HISTORY_DAYS_FOR_TRAIN, on_progress=_on_fetch_progress)
    log.info(f"LSTM 재학습: 데이터 수집 {len(history)}개 코인, {time.monotonic() - _fetch_started:.1f}초")

    _train_started = time.monotonic()
    for i, (coin, df) in enumerate(history.items(), start=1):
        set_progress("training", f"{coin} 모델 학습 중... ({i}/{len(history)})",
                     current_coin=coin, coin_index=i, coin_total=len(history), train_step="fitting")

        def _on_epoch(_coin, epoch_idx, total_epochs, _i=i):
            set_progress("training", f"{_coin} 모델 학습 중... epoch {epoch_idx}/{total_epochs}",
                         current_coin=_coin, coin_index=_i, coin_total=len(history), train_step="fitting")

        try:
            chain_length = lstm_model.get_finetune_chain_length(coin)
            force_full_retrain = chain_length >= FINETUNE_CHAIN_LENGTH_RESET_THRESHOLD
            if force_full_retrain:
                log.info(f"[retrain/{coin}] finetune_chain_length={chain_length} >= "
                         f"{FINETUNE_CHAIN_LENGTH_RESET_THRESHOLD} — 이번 슬롯은 전체 재학습"
                         f"(finetune=False)으로 전환해 가중치 고착을 방지합니다.")
            r = lstm_model.train_model(coin, df, epochs=8, history_days=HISTORY_DAYS_FOR_TRAIN,
                                        on_epoch=_on_epoch, finetune=not force_full_retrain)
        except Exception as e:
            # 한 코인 학습 실패가 이번 슬롯 전체(retrain_lstm() 전체 예외 전파 ->
            # last_retrain_at 미갱신 -> 다음 사이클에 90일 데이터 재수집+전체 재학습
            # 반복)로 번지지 않도록 코인 단위로 격리한다. run_one_cycle()의
            # 전략/코인 단위 예외 격리(analysis_team 리포트 2026-11-16_1422)와 동일한 패턴.
            log.exception(f"[retrain/{coin}] 학습 중 에러, 이 코인만 이번 슬롯 건너뜀: {e}")
            r = {
                "coin": coin,
                "trained": False,
                "reason": f"exception: {e}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
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

    # analysis_team 리포트 2026-11-21_2118(execution/synthesizer Top3 #1): results가
    # 전부 trained=False(예: 데이터 수집 전체 실패, 코인별 예외 전부 발생)여도
    # 여기서 무조건 last_lstm_train을 갱신하면 "실패인데 성공처럼" 기록되어
    # retrain_is_due()가 다음 슬롯(최대 2시간24분 후)까지 재시도를 막아버린다.
    # 실제로 하나라도 학습이 성공했을 때만 갱신해서, 전면 실패 시 다음 사이클
    # (1분 후)에 바로 재시도할 수 있게 한다.
    any_trained = any(r.get("trained") for r in results)
    if any_trained:
        with storage.connect() as conn:
            storage.set_meta(conn, "last_lstm_train", {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "results": results,
            })
    else:
        log.warning("이번 재학습 슬롯에서 코인이 전부 학습 실패 — last_lstm_train 메타를 갱신하지 않고 다음 사이클에 재시도합니다.")
    log.info(f"LSTM 재학습: 학습(fitting) 단계 {time.monotonic() - _train_started:.1f}초")
    set_progress("idle", "학습 완료, 대기 중")
    return any_trained


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
    # 예전엔 any(...)로 "코인 중 하나라도 모델이 있으면" 최초 학습을 건너뛰어서,
    # 특정 코인(예: ADA) 모델 파일만 손상/삭제돼도 그 코인은 다음 정기 재학습
    # 슬롯(최대 2시간24분 후)까지 계속 FileNotFoundError만 내고 있었다.
    # 코인별로 결손을 확인해, 하나라도 없으면 (retrain_lstm()이 COINS 전체를
    # 다시 학습하므로) 최초 학습을 수행한다 (analysis_team execution 리포트
    # 2026-11-16_1718 제안).
    missing_models = [c for c in COINS if not (lstm_model.MODEL_DIR / f"{c}_lstm.pt").exists()]
    if missing_models:
        log.info(f"저장된 LSTM 모델이 없는 코인: {missing_models} — 최초 학습을 먼저 수행합니다 (몇 분 걸릴 수 있음)...")
        try:
            retrain_lstm()  # 이 안에서 last_lstm_train 메타를 저장함 (내부에서 짧게 connect)
        except Exception as e:
            log.exception(f"최초 LSTM 학습 중 에러 (LSTM 없이 조건식 전략만으로 계속 진행): {e}")
        finally:
            _warned_missing_model.clear()  # 학습을 시도했으니 결손 상태가 바뀌었을 수 있어 경고 기록 초기화

    # 메모리 변수가 아니라 storage에 저장된 마지막 재학습 시각으로
    # 시작한다 — main_loop.py를 재시작해도 "이번 슬롯을 이미 했는지"를 안다.
    # (바로 위에서 최초 학습을 했다면 그 기록이 여기서 바로 "방금"으로 잡힌다.)
    # while 루프 안의 모든 storage.connect() 호출(사이클/재학습/prune)은 각자
    # try/except로 보호돼 있는데, 이 최초 조회는 보호가 없어서 DB 파일 손상이나
    # 디스크 I/O 에러가 나면 while 루프에 들어가기도 전에 프로세스 전체가 죽는
    # 문제가 있었다 (analysis_team execution 리포트 2026-11-22_0221). 실패해도
    # None으로 두고 계속 진행 — 최초 재학습 타이밍이 다소 늦어질 뿐(다음 사이클에
    # retrain_is_due(None)이 True가 되어 곧 재학습됨) 매매 자체는 영향받지 않는다.
    # 무한 재시도로 흐르지 않도록(매 사이클 반복 조회가 아니라) 여기서 최초
    # 1회만 수행하는 범위로 제한한다.
    try:
        with storage.connect() as conn:
            last_retrain_at = last_retrain_at_from_storage(conn)
            last_prune_date = last_predictions_prune_date_from_storage(conn)
    except Exception as e:
        log.exception(f"마지막 재학습/prune 기록 조회 중 에러 (초기값 없이 계속 진행): {e}")
        last_retrain_at = None
        last_prune_date = None
    if last_retrain_at:
        log.info(f"마지막 LSTM 재학습: {last_retrain_at} (저장된 기록 기준)")
    log.info(f"LSTM 재학습 스케줄: 매일 {', '.join(f'{h:02d}:{m:02d}' for h, m in RETRAIN_TIMES_LOCAL)} (하루 {len(RETRAIN_TIMES_LOCAL)}회)")

    while not _stop:
        _cycle_started = time.monotonic()  # 이번 사이클(수집+매매, 필요시 재학습) 소요시간 측정용
        try:
            with storage.connect() as conn:
                run_one_cycle(conn)
        except Exception as e:
            # 코인/전략 단위 예외(위 440행 근처)와 lstm 재학습(아래 _warned_missing_model
            # 초기화 부근)은 모두 "반복되는 예외는 1시간에 1회만 전체 트레이스" 쓰로틀이
            # 있는데, 이 사이클 최상위 except만 빠져 있었다. storage.connect() 자체가
            # 디스크 오류 등으로 계속 실패하는 가장 심각한 상황에서는 60초마다(하루 최대
            # 1440회) 전체 스택트레이스가 무한정 쌓일 수 있었다 — 이 파일 전체가 피하려
            # 한 로그 폭주가 가장 치명적인 장애 경로에서만 빠져 있던 역설 (analysis_team
            # execution 리포트 2026-11-22_0517 발견점 #2, synthesizer Top3 #2). 기존
            # _warned_repeated_exception 딕셔너리를 그대로 재사용하되, 사이클 전체는
            # 특정 (전략, 코인)에 묶이지 않으므로 키의 앞 두 자리를 "_cycle" 고정값으로
            # 채워 451행 패턴과 동일한 모양을 유지한다.
            repeat_key = ("_cycle", "_cycle", type(e).__name__)
            last_logged = _warned_repeated_exception.get(repeat_key)
            now_mono = time.monotonic()
            if last_logged is None or (now_mono - last_logged) >= _REPEATED_EXCEPTION_LOG_INTERVAL_SEC:
                log.exception(f"사이클 실행 중 에러 (스킵하고 계속): {e}")
                _warned_repeated_exception[repeat_key] = now_mono
            else:
                log.warning(f"사이클 실행 중 에러 반복 중 (같은 종류 "
                             f"{_REPEATED_EXCEPTION_LOG_INTERVAL_SEC}초 내 전체 트레이스는 "
                             f"이미 남겼음, 스킵하고 계속): {type(e).__name__}: {e}")

        now_local = datetime.now()
        # _elapsed(아래 650행 근처) 경고 로그가 재학습 때문에 느려진 사이클까지
        # 중복으로 찍지 않도록, 이번 반복에서 재학습을 시도했는지를 별도로
        # 기록한다. retrain_is_due(last_retrain_at)는 재학습 성공 시 last_retrain_at이
        # 바로 갱신돼 그 이후에는 "지금 당장 재학습이 필요한가"만 답할 뿐
        # "이번 반복에서 재학습을 했는가"를 알 수 없어 별도 플래그가 필요하다.
        _retrain_attempted_this_iteration = False
        # 재학습과 마찬가지로, 하루 1회 prune(아래 "today_str" 블록)도 순수 DB
        # 작업이지만 650행 근처의 _elapsed 측정 구간 안에서 실행되므로, 그날의
        # 한 사이클만 "느려지는 조짐"처럼 보여 거짓 경고를 낼 수 있다. prune은
        # 네트워크 호출이 없어 실제 드리프트(네트워크 지연 등) 신호를 가릴
        # 위험은 낮다 (analysis_team execution 리포트 2026-11-22_0318, synthesizer
        # Top3 #3).
        _prune_attempted_this_iteration = False
        if retrain_is_due(last_retrain_at):
            _retrain_attempted_this_iteration = True
            # 재학습 동안 run_one_cycle()이 동기적으로 멈추는 공백 구간의 실제 길이를
            # run.log에서 가늠할 수 있도록 소요 시간을 기록한다.
            # (analysis_team 리포트 2026-11-16_1521_execution.md)
            _retrain_started = time.monotonic()
            try:
                # retrain_lstm()이 내부에서 "하나라도 학습 성공" 여부를 반환한다
                # (analysis_team 리포트 2026-11-21_2118). 전부 실패했으면 이번
                # 슬롯을 "완료"로 치지 않고 last_retrain_at을 그대로 둬서, 다음
                # 사이클(1분 후)에 retrain_is_due()가 다시 True가 되게 한다.
                any_trained = retrain_lstm()  # 내부에서 단계별로 짧게 connect — 재학습 도는 동안 DB를 안 잡음
                if any_trained:
                    last_retrain_at = now_local
                else:
                    log.warning("이번 재학습 슬롯 전체 실패 — last_retrain_at을 갱신하지 않아 다음 사이클에 재시도합니다.")
            except Exception as e:
                log.exception(f"LSTM 재학습 중 에러: {e}")
            finally:
                log.info(f"LSTM 재학습 소요 시간: {time.monotonic() - _retrain_started:.1f}초 "
                          f"(이 구간 동안 매매 사이클은 멈춰 있었음)")
                _warned_missing_model.clear()  # 재학습을 시도했으니 결손 상태가 바뀌었을 수 있어 경고 기록 초기화

        today_str = now_local.strftime("%Y-%m-%d")
        if last_prune_date != today_str:
            # 하루 1회만(날짜가 바뀐 첫 사이클에서) predictions 보관주기 정리를
            # 실행한다. 거래/손절 판단에 쓰는 로직은 전혀 건드리지 않고,
            # 이미 확정된(resolved=1) 오래된 예측 기록만 지운다.
            _prune_attempted_this_iteration = True
            try:
                with storage.connect() as conn:
                    n_pruned = storage.prune_predictions(conn, keep_days=PREDICTIONS_PRUNE_KEEP_DAYS)
                    # predictions와 동일한 비일관성(append-only 무한 누적)이 snapshots에도
                    # 있어 같은 유지보수 시점에 같이 정리한다 (analysis_team 리포트
                    # 2026-11-21_2221_storage.md).
                    n_pruned_snapshots = storage.prune_snapshots(conn, keep_days=SNAPSHOTS_PRUNE_KEEP_DAYS)
                    # model_trainings도 predictions/snapshots와 같은 유지보수 시점에
                    # 함께 정리한다 (analysis_team 리포트 2026-11-22_0221_storage.md,
                    # synthesizer Top3 #1). trained=0(실패 기록)만 대상이라 모델
                    # 성능 추이(trained=1)는 전혀 영향받지 않는다.
                    n_pruned_trainings = storage.prune_model_trainings(conn, keep_days=MODEL_TRAININGS_PRUNE_KEEP_DAYS)
                    # WAL 파일도 함께 줄인다 — 다른 커넥션이 몰려 있으면 부분적으로만
                    # 체크포인트되고 예외 없이 끝나므로 트레이딩 사이클에 안전하다.
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    storage.set_meta(conn, "last_predictions_prune", {
                        "date": today_str,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "rows_deleted": n_pruned,
                        "snapshots_rows_deleted": n_pruned_snapshots,
                        "model_trainings_rows_deleted": n_pruned_trainings,
                    })
                last_prune_date = today_str
                log.info(f"predictions 보관주기 정리 완료: {n_pruned}건 삭제 (keep_days={PREDICTIONS_PRUNE_KEEP_DAYS}), "
                         f"snapshots {n_pruned_snapshots}건 삭제 (keep_days={SNAPSHOTS_PRUNE_KEEP_DAYS}), "
                         f"model_trainings {n_pruned_trainings}건 삭제 (keep_days={MODEL_TRAININGS_PRUNE_KEEP_DAYS}, trained=0만 대상)")
            except Exception as e:
                log.exception(f"predictions/snapshots 보관주기 정리 중 에러 (다음 사이클에 재시도): {e}")

        retrain_left = _fmt_remaining(time_until_next_retrain(last_retrain_at))
        analysis_left = _fmt_remaining(time_until_next_cloud_analysis())
        log.info(f"다음 LSTM 재학습까지 {retrain_left} | 다음 클라우드 분석팀 실행까지 {analysis_left}")

        # 고정 "60번 sleep(1)"이었을 때는 실제 주기가 "이번 사이클 작업시간(수집+매매,
        # 필요시 재학습) + 60초"가 되어, 평균 처리시간이 몇 초만 돼도 하루 1440사이클
        # 누적 시 분~시간 단위로 벌어졌다(사이클이 죽는 건 아니고 서서히 밀리는 것).
        # 이번 사이클 시작부터 지금까지 걸린 시간만큼 빼서 60초 주기에 맞춘다 —
        # 재학습이 섞인 사이클처럼 이미 60초를 넘겼으면 남는 시간이 0이라 바로 다음
        # 사이클로 넘어간다 (analysis_team execution 리포트 2026-11-16_1718 제안).
        _elapsed = time.monotonic() - _cycle_started
        # 재학습이 섞이지 않은 평상시 사이클인데도 임계치를 넘겼다면, 재학습과
        # 무관한 원인(네트워크 지연 누적, 전략/코인 수 증가 등)으로 처리시간이
        # 60초 주기에 근접하고 있다는 조짐이라 경고를 남긴다 (analysis_team
        # execution 리포트 2026-11-22_0018, synthesizer Top3 #1). 재학습이 있었던
        # 사이클은 610행 근처에서 이미 소요시간을 별도로 로깅하므로 중복 경고를
        # 피하기 위해 여기서는 제외한다.
        if (not _retrain_attempted_this_iteration and not _prune_attempted_this_iteration
                and _elapsed > CYCLE_ELAPSED_WARN_THRESHOLD_SEC):
            log.warning(f"사이클 처리에 {_elapsed:.1f}초 소요(60초 중) — "
                        f"{CYCLE_ELAPSED_WARN_THRESHOLD_SEC}초 임계치 초과, 60초 주기에 근접")
        _sleep_remaining = max(0.0, CYCLE_INTERVAL_SEC - _elapsed)
        _slept = 0.0
        while _slept < _sleep_remaining:
            if _stop:
                break
            chunk = min(1.0, _sleep_remaining - _slept)
            time.sleep(chunk)
            _slept += chunk

    log.info("=== 루프 정상 종료 ===")


if __name__ == "__main__":
    main()
