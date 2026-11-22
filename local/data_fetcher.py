"""
Upbit 공개 API로 1분봉 시세를 받아오는 모듈 (PC 상주 실행용).

클라우드 버전(상위 디렉토리의 data_fetcher.py)과 달리, 이 모듈은
"지금 이 순간의 1분봉"과 "LSTM 학습용 최근 N일치 1분봉"만 다룬다.
1분봉은 하루에만 1,440개라 상장일부터 전체를 받는 건 비현실적이므로
(Upbit이 페이지당 200개 제한), 학습 데이터 기간은 fetch_minute_history의
days 인자로 조절한다 (기본 14일 = 약 20,000개 캔들, 노트북 CPU로도
LSTM 학습 가능한 크기).

Upbit API는 분당 요청수 제한이 있으므로 sleep_sec으로 매너를 지킨다.
"""

from __future__ import annotations

import http.client
import json
import logging
import threading
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data_cache"
DATA_DIR.mkdir(exist_ok=True)

UPBIT_BASE = "https://api.upbit.com"

COINS = ["BTC", "ETH", "XRP", "SOL", "DOGE", "ADA"]

UPBIT_MARKETS = {
    "BTC": "KRW-BTC",
    "ETH": "KRW-ETH",
    "XRP": "KRW-XRP",
    "SOL": "KRW-SOL",
    "DOGE": "KRW-DOGE",
    "ADA": "KRW-ADA",
}


class FetchError(Exception):
    pass


# main_loop.py 110~130행 근처에 이미 있는 "반복되는 예외는 N초마다 1회만 전체
# 트레이스" 패턴을 data_fetcher.py에도 적용한다. fetch_all_latest_candles/
# fetch_all_minute_history의 _fetch_one은 main_loop.py에서 매 사이클(60초)
# 마다 호출되는데, 특정 코인 마켓이 한동안 계속 실패하면(Upbit 점검, 네트워크
# 문제 등) 지금까지는 실패마다 매번 전체 스택트레이스를 남겨 run.log가
# 금방 폭주할 수 있었다 (analysis_team execution 리포트 2026-11-22_0118,
# 2026-11-22_0221). main_loop.py 쪽은 (전략, 코인, 예외타입)이 키지만 이
# 모듈은 전략이 없으므로 (코인, 예외타입)만으로 구분한다.
_warned_repeated_exception: dict[tuple[str, str], float] = {}
_REPEATED_EXCEPTION_LOG_INTERVAL_SEC = 3600  # 1시간마다 전체 트레이스 1회


def _log_fetch_exception(log_prefix: str, coin: str, e: Exception) -> None:
    """실패마다 log.exception(전체 스택트레이스)을 남기는 대신, 같은
    (코인, 예외타입) 조합은 1시간에 1회만 전체 트레이스를 남기고 그 사이엔
    log.warning만 남긴다. 실제로 장애가 지속되는 걸 운영자가 너무 늦게
    알아차리지 않도록 "완전 무음화"가 아니라 "일정 시간마다 1회"로
    완화한다(main_loop.py의 동일 패턴과 같은 트레이드오프)."""
    key = (coin, type(e).__name__)
    now_mono = time.monotonic()
    last_logged = _warned_repeated_exception.get(key)
    if last_logged is None or (now_mono - last_logged) >= _REPEATED_EXCEPTION_LOG_INTERVAL_SEC:
        log.exception(f"{log_prefix}: {e}")
        _warned_repeated_exception[key] = now_mono
    else:
        log.warning(f"{log_prefix} (반복 중, 같은 종류 {_REPEATED_EXCEPTION_LOG_INTERVAL_SEC}초 내 "
                    f"전체 트레이스는 이미 남겼음): {type(e).__name__}: {e}")


class _RateLimiter:
    """여러 스레드가 동시에 호출해도 전체 요청 간 최소 간격(min_interval_sec)을
    전역으로 강제하는 간단한 레이트 리미터.

    fetch_all_minute_history는 6코인을 각자 스레드로 동시에 돌리는데, 각
    스레드가 "자기 혼자" sleep_sec(기본 0.12초)만 지키면 6개 스레드가 각자
    독립적으로 0.12초마다 요청을 쏘게 되어 전체 초당 요청 수는 순차 실행
    (~8.3req/s) 대비 최대 코인 수(6)배(~50req/s)까지 늘어날 수 있다 — 코드
    주석은 "병렬화해도 교류량 자체는 늘지 않는다"고 돼 있었지만 실제로는
    그렇지 않았다. 이 리미터를 모든 코인 스레드가 공유하면, 스레드 수와
    무관하게 Upbit에 가는 실제 요청 속도가 순차 실행과 동일하게 유지된다."""

    def __init__(self, min_interval_sec: float):
        self._min_interval = min_interval_sec
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait_sec = self._next_allowed - now
            if wait_sec > 0:
                time.sleep(wait_sec)
                now = time.monotonic()
            self._next_allowed = max(now, self._next_allowed) + self._min_interval


def _http_get_json(url: str, timeout: int = 10, retries: int = 3) -> dict | list:
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "crypto-sim-local/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                json.JSONDecodeError, UnicodeDecodeError,
                ConnectionError, http.client.HTTPException) as e:
            # JSONDecodeError/UnicodeDecodeError: Upbit이 손상되거나 빈 응답,
            # 혹은 HTML 에러 페이지를 반환하는 경우가 드물게 있는데, 예전에는
            # 이 두 예외가 여기서 안 잡혀서 재시도 없이 바로 호출부까지
            # 전파되며 프로그램이 죽을 수 있었다. 다른 네트워크 에러와 동일하게
            # 재시도 대상으로 묶는다.
            # ConnectionError(ConnectionResetError/BrokenPipeError 등)와
            # http.client.HTTPException(IncompleteRead/RemoteDisconnected 등)은
            # "연결 수립"이 아니라 "응답 본문 수신 중" 끊김이라 기존 튜플에
            # 빠져 있었다. 이 경우 여기서 재시도되지 않고, fetch_minute_history의
            # "80% 이상 모았으면 부분 데이터로 반환" 최적화(위 FetchError
            # except 블록)도 건너뛴 채 fetch_all_minute_history의 broad
            # except Exception까지 그대로 전파되고 있었다 (analysis_team
            # execution 리포트 2026-11-22_0517 발견점 #1, synthesizer Top3 #1).
            last_err = e
            if isinstance(e, urllib.error.HTTPError) and e.code != 429 and not (500 <= e.code < 600):
                # 영구적인 4xx(마켓 코드 오타, API 스펙 변경 등)는 재시도해도
                # 결과가 달라지지 않는데, 기존에는 이런 경우도 네트워크성 에러와
                # 동일하게 최대 3회(백오프 1+2초 포함)까지 재시도한 뒤에야
                # 실패했다. 매 60초 사이클마다 호출되는
                # fetch_latest_minute_candles에서 이런 상황이 생기면 그만큼
                # 사이클 지연에 조금씩 기여할 수 있었다(analysis_team execution
                # 리포트 2026-11-22_0918 구체적 제안 #2). 429(rate limit)와
                # 5xx(서버 쪽 일시 장애)만 재시도 대상으로 남기고, 그 외 4xx는
                # 즉시 실패로 전환한다.
                raise FetchError(f"GET failed with non-retryable HTTP {e.code}: {url} ({e})") from e
            # 마지막 시도(attempt == retries - 1)가 실패하면 그 뒤로는 어차피
            # 재시도하지 않고 바로 루프를 빠져나가 FetchError를 던지므로, 그
            # 직전에 sleep할 이유가 없다. 예전에는 마지막 시도 실패 후에도
            # sleep해서, 특히 매 60초 사이클마다 호출되는
            # fetch_latest_minute_candles(timeout=5, retries=2)에서 완전
            # 실패까지 걸리는 시간에 불필요한 지연이 매번 더해졌다
            # (analysis_team execution 리포트 2026-11-22_0718 발견점 #2).
            if attempt < retries - 1:
                sleep_sec = 1.0 * (attempt + 1)
                if isinstance(e, urllib.error.HTTPError) and e.code == 429:
                    # Upbit이 레이트리밋(429) 응답에 Retry-After 헤더로 다음
                    # 요청까지 기다려야 할 시간(초)을 알려주는 경우, 고정
                    # 백오프(1초, 2초)보다 이 값을 우선한다 — 서버가 요구하는
                    # 시간보다 일찍 재시도해 추가로 429를 받는 상황을 줄인다
                    # (analysis_team execution 리포트 2026-11-22_1018 발견점
                    # #2). 다만 60초 사이클마다 호출되는
                    # fetch_latest_minute_candles(timeout=5, retries=2)가
                    # 이 때문에 사이클 드리프트 경고 임계값(30초)을 넘길
                    # 정도로 오래 멈추면 안 되므로 상한(5초)을 둔다. 헤더가
                    # 없거나 값 파싱이 실패하면 기존 고정 백오프로 폴백한다.
                    retry_after = e.headers.get("Retry-After") if e.headers else None
                    if retry_after is not None:
                        try:
                            sleep_sec = min(float(retry_after), 5.0)
                        except (TypeError, ValueError):
                            pass
                time.sleep(sleep_sec)
    raise FetchError(f"GET failed after {retries} retries: {url} ({last_err})")


def _parse_minute_df(raw: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(raw)
    df = df.rename(columns={
        "candle_date_time_kst": "open_time",
        "opening_price": "open",
        "high_price": "high",
        "low_price": "low",
        "trade_price": "close",
        "candle_acc_trade_volume": "volume",
    })
    df["open_time"] = pd.to_datetime(df["open_time"])
    df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
    return df[["open_time", "open", "high", "low", "close", "volume"]]


def fetch_latest_minute_candles(market: str, count: int = 200,
                                 timeout: int = 5, retries: int = 2,
                                 rate_limiter: "_RateLimiter | None" = None) -> pd.DataFrame:
    """가장 최신 1분봉 count개 (실시간 신호 계산용). 마지막 행이 가장 최근 캔들.

    기본 timeout/retries는 _http_get_json()의 기본값(10/3, fetch_minute_history처럼
    하루 10회만 호출되는 경로용)보다 짧게 잡았다 — 이 함수는 fetch_all_latest_candles()를
    통해 60초 주기 매매 사이클마다 호출되므로, 한 코인이 느려지면 최대 10초*3회+백오프
    ≈33초까지 걸려 CYCLE_ELAPSED_WARN_THRESHOLD_SEC(30초) 경고를 유발할 수 있었다
    (analysis_team execution 리포트 2026-11-22_0318, synthesizer Top3 #1).

    rate_limiter: 주어지면 요청 전에 이걸로 전역 페이싱을 한다 — fetch_minute_history와
    같은 패턴으로, fetch_all_latest_candles()가 6코인 스레드가 공유하는 리미터를
    넘겨서 전체 요청 속도를 제어하는 용도. None(기본값)이면 예전과 동일하게
    페이싱 없이 바로 요청한다(단독 호출/테스트 동작 그대로 유지).
    """
    if rate_limiter is not None:
        rate_limiter.wait()
    # Upbit 분봉 API는 단일 요청(페이지네이션 없음)당 최대 200개까지만
    # 돌려주며, 모듈 docstring도 이 제한을 명시한다. 이 함수를 호출하는
    # main_loop.py(fetch_all_latest_candles(count=max(lstm_model.SEQ_LEN + 5, 200)))가
    # count를 넘기는데, 지금은 SEQ_LEN(60)+5=65로 200보다 작아 문제가 없지만
    # 코드 어디에도 상한을 강제/경고하는 장치가 없어 향후 SEQ_LEN이 늘어나면
    # Upbit이 조용히 200개로 잘라 응답하는 "캔들 부족" 경고가 영원히 반복되는
    # 조용한 회귀가 생길 수 있었다 (analysis_team execution 리포트
    # 2026-11-22_1219 발견점 #2, synthesizer Top3 #1). 호출자가 몰라도
    # 안전하도록 피호출부에서 클램프하고, 잘렸을 때만 1회성 디버그 로그를 남긴다.
    if count > 200:
        log.debug(f"{market} count={count}가 Upbit 단일 요청 상한(200)을 초과해 200으로 조정")
        count = 200
    url = f"{UPBIT_BASE}/v1/candles/minutes/1?market={market}&count={count}"
    raw = _http_get_json(url, timeout=timeout, retries=retries)
    if not raw:
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])
    return _parse_minute_df(raw)


def fetch_minute_history(market: str, days: int = 14, sleep_sec: float = 0.12,
                          max_pages: int | None = None, on_page=None,
                          rate_limiter: "_RateLimiter | None" = None,
                          min_time: "pd.Timestamp | None" = None) -> tuple[pd.DataFrame, bool]:
    """최근 `days`일치 1분봉 전체를 페이지네이션으로 수집 (LSTM 학습용).
    Upbit은 페이지당 최대 200개(=200분)만 주므로, days=14면 약 14*1440/200≈101회 호출.
    `to` 파라미터로 과거 방향 페이지네이션.

    반환값은 (df, is_partial) 튜플이다. is_partial=True는 페이지 실패(80% 이상
    모은 뒤 포기) 또는 max_pages 도달로 목표(`days`일치) 전체를 다 모으지
    못하고 중간에 멈췄다는 뜻이다 — 호출부(fetch_all_minute_history의
    _fetch_one)가 이 경우 "오늘 전체 재수집 완료" 마커(last_full_refresh)를
    기록하지 않도록 구분하는 용도다. 이전에는 이 둘을 구분할 방법이 없어서,
    부분 성공도 완료와 동일하게 마커에 기록되어 그 코인이 당일 남은 재학습
    슬롯 동안 계속 증분 수집만 하며 학습 데이터 창이 줄어든 채 유지되는
    문제가 있었다 (analysis_team execution 리포트 2026-11-22_1418.md 발견점
    #1). all_rows가 비어 완전히 빈 DataFrame을 반환하는 경우(거래소에 더 받을
    데이터가 없는 정상 상황)는 is_partial=False다 — "부분 수집"이 아니라
    "받을 게 없어서 끝"이기 때문이다.

    min_time: 주어지면(이미 캐시에 min_time까지의 데이터가 있는 경우), 기존
    `cutoff`(now - days) 대신 "min_time과 cutoff 중 더 최근 시점"을 조기 중단
    기준으로 써서 캐시에 없는 새 데이터만 받아온다. 재학습마다 `days`일치
    전체를 매번 통째로 재수집해 API 호출이 과도하게 누적되는 문제
    (analysis_team execution 리포트 2026-11-22_0918 발견점 #1, synthesizer
    Top3 #1)를 줄이기 위한 증분 수집용 — fetch_all_minute_history()가 캐시의
    마지막 시점을 넘겨준다. None(기본값)이면 기존과 완전히 동일하게 동작
    (단독 호출/테스트 동작 그대로 유지).

    max_pages 기본값은 None이면 필요 페이지 수(days*1440/200)에 20% 여유를 더해
    자동 계산한다 (예: 30일 -> 216 * 1.2 ≈ 259). 이렇게 상한을 현실적으로 두는 이유:
    예전에 상한이 1000으로 고정돼 있을 때, Upbit이 같은 시점을 반복 응답하는 등
    `to` 파라미터가 실질적으로 전진하지 않는 드문 경우 break 조건을 못 만족해서
    최악의 경우 1000페이지 전체를 돌며 몇 시간씩 멈춘 것처럼 보이는 버그가 있었음
    ("어쩔 때는 멈추고 어쩔 때는 그냥 돌아간다"는 바로 이 증상).

    on_page(page_idx, oldest_time, rows_so_far): 페이지 하나 받을 때마다 호출
    (진행 로그/진행률 표시용, 선택). 멈춘 건지 그냥 느린 건지 구분하려면 필수.

    rate_limiter: 주어지면 각 페이지 요청 전에 이걸로 전역 페이싱을 하고,
    기존 sleep_sec 기반의 "이 호출 혼자"만의 간격 유지는 건너뛴다 (여러
    코인을 동시에 fetch_minute_history로 돌릴 때 fetch_all_minute_history가
    공유 리미터를 넘겨서 전체 요청 속도를 제어하는 용도). None(기본값)이면
    예전과 동일하게 sleep_sec만으로 이 호출 단독의 페이싱을 한다 — 단독
    호출(예: 테스트, 일회성 스크립트)의 동작은 그대로 유지된다."""
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    # min_time이 cutoff보다 더 최근이면(=캐시가 이미 그 시점까지 있으면) 그
    # 시점에서 조기 중단해, 캐시에 이미 있는 과거 페이지들은 다시 받지 않는다.
    effective_cutoff = cutoff if min_time is None or min_time <= cutoff else min_time
    if max_pages is None:
        max_pages = int(days * 1440 / 200 * 1.2) + 5
    # 목표 페이지 수(마진 20% 포함 전) 대비 80% 이상을 이미 모은 상태에서
    # 한 페이지가 실패하면, 전체를 버리지 않고 그때까지 모은 부분 데이터로
    # 정상 반환한다 (analysis_team 리포트 2026-11-21_2221_execution.md).
    # 그 이전에(너무 적게 모은 상태에서) 실패하면 기존과 동일하게 예외를
    # 그대로 전파해 fetch_all_minute_history()의 캐시 폴백 경로를 그대로 탄다.
    _expected_pages = days * 1440 / 200
    _min_pages_for_partial = int(_expected_pages * 0.8)
    all_rows: list[dict] = []
    to_param = ""
    prev_to_param = None
    stuck_count = 0
    is_partial = False

    for page_idx in range(1, max_pages + 1):
        if rate_limiter is not None:
            rate_limiter.wait()
        url = f"{UPBIT_BASE}/v1/candles/minutes/1?market={market}&count=200"
        if to_param:
            url += f"&to={to_param}"
        try:
            raw = _http_get_json(url)
        except FetchError:
            # 예전에는 이 예외가 그대로 fetch_minute_history() 밖으로 전파되어,
            # 650페이지 중 649페이지까지 모아둔 all_rows까지 통째로 버려지고
            # fetch_all_minute_history()가 디스크 캐시(최대 2시간24분 전)로
            # 전체 대체했다. 이미 충분히(목표의 80% 이상) 모았다면 그 부분
            # 데이터를 그대로 쓰는 쪽이 낫다.
            if page_idx - 1 >= _min_pages_for_partial:
                log.warning(
                    f"{market} 페이지 {page_idx}/{max_pages} 수집 중 실패했지만 "
                    f"이미 {page_idx - 1}페이지(목표 ~{_expected_pages:.0f}페이지의 "
                    f"{(page_idx - 1) / _expected_pages:.0%}) 모았으므로 부분 데이터로 반환합니다."
                )
                is_partial = True
                break
            raise
        if not raw:
            break
        all_rows.extend(raw)
        oldest = raw[-1]
        oldest_time = pd.to_datetime(oldest["candle_date_time_kst"])
        new_to_param = oldest["candle_date_time_utc"]

        if on_page:
            on_page(page_idx, oldest_time, len(all_rows))

        # 진행이 전혀 안 되는 경우(같은 to 파라미터 반복) 감지 — 멈춘 것처럼
        # 보이는 무한 루프를 조기에 끊어낸다. 1~2회는 API 쪽의 일시적 중복
        # 응답일 수 있으니 바로 끊지 않고 몇 번 더 참아본 뒤 포기한다.
        if new_to_param == prev_to_param:
            stuck_count += 1
            if stuck_count >= 3:
                # 과거엔 print()만 썼는데, main_loop.py의 logging.basicConfig는
                # print() 출력을 전혀 거치지 않아 이 경고가 run.log에는 안 남고
                # 콘솔에만 찍혔다(analysis_team 리포트 2026-11-16_1619_execution.md).
                # 콘솔 없이 서비스로 상주 실행 중이면 이 경고를 영영 못 보게 되므로
                # log.warning으로 교체해 run.log에도 남긴다.
                log.warning(f"{market} 페이지네이션이 {new_to_param} 시점에서 "
                            f"전진하지 않아 중단 (수집된 {len(all_rows)}개로 계속)")
                break
        else:
            stuck_count = 0
        prev_to_param = new_to_param
        to_param = new_to_param

        if oldest_time <= effective_cutoff or len(raw) < 200:
            break
        if rate_limiter is None:
            time.sleep(sleep_sec)
    else:
        # 위와 동일한 이유로 log.warning 사용 (run.log에 남기기 위함).
        log.warning(f"{market} 수집이 max_pages({max_pages})에 도달해 중단됨 "
                    f"(목표 {days}일 중 일부만 수집됐을 수 있음)")
        is_partial = True

    if not all_rows:
        # 받을 데이터가 전혀 없었던(거래소 쪽에 더 없는) 정상 상황이라 부분
        # 수집으로 취급하지 않는다 — 위 두 분기(페이지 실패/max_pages 도달)와는
        # 다르게 "목표를 못 채우고 멈춘" 게 아니라 "처음부터 받을 게 없었던" 경우.
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"]), False

    df = _parse_minute_df(all_rows)
    return df[df["open_time"] >= effective_cutoff].reset_index(drop=True), is_partial


def fetch_all_latest_candles(count: int = 200) -> dict[str, pd.DataFrame]:
    """6대 코인의 최신 1분봉 (매분 실행되는 메인 루프에서 호출). 코인당 호출이
    1번뿐이라 원래도 느리진 않지만, 순차 대신 동시에 보내서 사이클 자체를
    조금이라도 더 짧게 만든다.

    fetch_all_minute_history()는 6코인 스레드가 전역 _RateLimiter를 공유해
    Upbit 호출 속도를 명시적으로 제어하는데, 이 함수는 그런 보호 없이 6개
    요청을 완전히 동시에 쏘고 있었다(analysis_team execution 리포트
    2026-11-22_0718 발견점 #3) — 두 경로의 설계가 비대칭적이었다. 여기서도
    같은 _RateLimiter를 적용해 설계를 통일한다. 다만 이 경로는 60초마다 호출되고
    코인 수가 6개뿐이라 history_rate_limiter와 같은 0.12초 간격을 쓰면 최악의
    경우 6*0.12=0.72초가 추가로 걸려 CYCLE_ELAPSED_WARN_THRESHOLD_SEC(30초)
    경고 여유를 줄일 수 있으므로, 간격을 충분히 짧게(0.02초) 잡아 Upbit
    순간 동시 요청 수만 완화하고 전체 사이클 길이에는 사실상 영향을 주지
    않도록 한다.
    """
    result: dict[str, pd.DataFrame] = {}
    latest_rate_limiter = _RateLimiter(min_interval_sec=0.02)

    def _fetch_one(coin: str) -> tuple[str, pd.DataFrame | None]:
        try:
            return coin, fetch_latest_minute_candles(UPBIT_MARKETS[coin], count=count,
                                                       rate_limiter=latest_rate_limiter)
        except Exception as e:
            # FetchError(네트워크/HTTP 재시도 소진)뿐 아니라, Upbit 응답 형식이
            # 예상과 달라 _parse_minute_df 등에서 터지는 KeyError류까지 전부 잡아서
            # 코인 1개의 오류가 ThreadPoolExecutor를 통해 전파되어 6코인 전체
            # 수집(fetch_all_latest_candles)을 실패시키지 않도록 한다. 반복되는
            # 실패는 _log_fetch_exception()이 1시간에 1회만 전체 스택트레이스를
            # 남기고 그 사이엔 warning만 남긴다(원인 진단은 유지하면서 로그
            # 폭주를 막음).
            _log_fetch_exception(f"[WARN] {coin} 최신 캔들 수집 실패", coin, e)
            return coin, None

    with ThreadPoolExecutor(max_workers=len(COINS)) as executor:
        for future in as_completed(executor.submit(_fetch_one, coin) for coin in COINS):
            coin, df = future.result()
            if df is not None:
                result[coin] = df
    return result


# 재학습(main_loop.py 기준 하루 10회)마다 fetch_all_minute_history가 매번
# HISTORY_DAYS_FOR_TRAIN(90)일치 전체를 통째로 재수집해, 직전 슬롯과 거의
# 겹치는 데이터까지 코인당 약 648페이지 × 6코인 ≈ 3,900회의 Upbit 호출이
# 2시간24분마다 반복되는 문제가 있었다 (analysis_team execution 리포트
# 2026-11-22_0918 발견점 #1, synthesizer Top3 #1). 아래 두 전역은 "하루에 한
# 번은 전체 재수집하되, 그 사이에는 캐시의 마지막 시점 이후분만 받는다"는
# 안전장치를 구현하기 위한 상태다 — 증분 수집만 계속하면 Upbit이 과거
# 데이터를 드물게 사후 정정하는 경우를 영영 반영하지 못할 수 있어(execution
# 리포트 자체가 지적한 트레이드오프), 날짜가 바뀌면(로컬 날짜 기준) 코인별로
# 1회씩 강제로 전체 재수집하도록 한다.
_FULL_REFRESH_MARKER_PATH = DATA_DIR / "last_full_refresh.json"


def _load_last_full_refresh() -> dict[str, str]:
    if not _FULL_REFRESH_MARKER_PATH.exists():
        return {}
    try:
        return json.loads(_FULL_REFRESH_MARKER_PATH.read_text())
    except Exception as e:
        # 마커 파일이 없거나 손상된 경우 "오늘 아직 전체 재수집 안 함"으로
        # 취급 — 이번 회차는 전체 코인이 전체 재수집으로 폴백되어 조금
        # 느려지지만(기존 동작과 동일한 비용), 데이터가 틀리게 섞이는 일은 없다.
        log.warning(f"last_full_refresh.json 읽기 실패, 이번 회차는 전체 재수집으로 폴백: {e}")
        return {}


def _save_last_full_refresh(marker: dict[str, str]) -> None:
    try:
        _FULL_REFRESH_MARKER_PATH.write_text(json.dumps(marker, ensure_ascii=False))
    except Exception as e:
        # 쓰기 실패해도 다음 재학습 때 다시 "오늘 전체 재수집 안 함"으로 보고
        # 재시도하므로 치명적이지 않다 — 그냥 그 다음 회차도 전체 재수집이
        # 될 뿐 (에러 전파 X).
        log.warning(f"last_full_refresh.json 쓰기 실패 (다음 회차에 재시도): {e}")


def fetch_all_minute_history(days: int = 14, cache: bool = True, on_progress=None) -> dict[str, pd.DataFrame]:
    """6대 코인의 최근 N일 1분봉 전체 (LSTM 학습/재학습 시 호출). 디스크 캐시 지원.

    6개 코인을 각자 별도 스레드에서 동시에 수집한다 (예전엔 순차로 하나씩
    받아서 6배 느렸음 — 코인당 ~90초면 전체 ~9분이던 게, 병렬화하면 전체도
    ~90초 선으로 줄어듦). 모든 코인 스레드가 전역 `_RateLimiter` 하나를
    공유해서, 실제 Upbit에 가는 요청 간격이 "전체 기준" sleep_sec(기본
    0.12초)로 유지된다 — 예전에는 각 스레드가 "자기 혼자" sleep_sec만
    지켰기 때문에 6개 스레드가 각자 0.12초마다 쏘면 전체 초당 요청 수가
    순차 실행 대비 최대 6배까지 늘어나는 문제가 있었다(주석은 "늘지 않는다"고
    돼 있었지만 실제 동작은 그렇지 않았다). 공유 리미터로 이제는 스레드 수와
    무관하게 전체 요청 속도가 순차 실행과 같게 유지된다.

    캐시 CSV가 있고 오늘 그 코인을 아직 전체 재수집하지 않았다면, 캐시의
    마지막 open_time 이후분만 받아서 병합한다(증분 수집, 위 모듈 docstring
    참고). 코인별로 로컬 날짜가 바뀐 뒤 처음 호출될 때는(또는 캐시/마커가
    없거나 읽기 실패하면) 안전하게 전체 재수집으로 폴백한다.

    on_progress(coin, index, total): 코인 하나 수집이 시작/진행될 때마다
    호출 (대시보드 진행표시 + 워치독 하트비트용). 여러 스레드에서 동시에
    호출될 수 있으므로 호출 자체를 락으로 감싼다 — set_progress가 매번
    짧게 DB connect를 여는 구조라 동시 호출돼도 storage.py의 재시도
    로직으로 안전하지만, 로그 출력 순서를 깔끔하게 유지하기 위해 직렬화한다."""
    expected_pages = int(days * 1440 / 200)
    result: dict[str, pd.DataFrame] = {}
    print_lock = threading.Lock()
    # 6개 코인 스레드가 모두 이 리미터 하나를 공유 — 전체 요청 속도를
    # 스레드 수와 무관하게 "코인 1개 기준" 간격(0.12초)으로 제한한다.
    history_rate_limiter = _RateLimiter(min_interval_sec=0.12)

    # 마커는 스레드 풀 시작 "전"에 한 번만 읽고, 끝난 "후"에 한 번만 쓴다 —
    # 6개 스레드가 각자 파일을 읽고/쓰면 경합(동시 읽기-수정-쓰기로 서로의
    # 갱신을 덮어씀)이 생길 수 있으므로, 읽기/쓰기 자체를 스레드 풀 바깥에서
    # 직렬화해 완전히 피한다.
    today_str = str(pd.Timestamp.now().date())
    last_full_refresh = _load_last_full_refresh()

    def _fetch_one(coin: str, i: int) -> tuple[str, pd.DataFrame | None, bool]:
        if on_progress:
            with print_lock:
                on_progress(coin, i, len(COINS))

        def _on_page(page_idx, oldest_time, rows_so_far, _coin=coin, _i=i):
            # 10페이지마다(약 1~2초 간격) 한 번씩만 찍어서 로그 폭주를 막되,
            # "멈춘 건지 그냥 느린 건지"는 바로 구분할 수 있게 한다.
            if page_idx % 10 == 0 or page_idx == 1:
                with print_lock:
                    # 다른 경고들과 패턴을 맞춰 log.info로 — run.log에서 진행상황을
                    # 그대로 확인할 수 있게 한다 (analysis_team 2026-11-16_1619_execution.md).
                    log.info(f"[{_coin}] 페이지 {page_idx}/~{expected_pages} 수집 중... "
                             f"(누적 {rows_so_far}개, 가장 오래된 시점 {oldest_time})")
                    if on_progress:
                        on_progress(_coin, _i, len(COINS))

        cache_path = DATA_DIR / f"{coin}_1m_{days}d.csv"
        force_full = last_full_refresh.get(coin) != today_str
        cached_df: pd.DataFrame | None = None
        min_time = None
        if not force_full and cache and cache_path.exists():
            try:
                cached_df = pd.read_csv(cache_path, parse_dates=["open_time"])
                if len(cached_df):
                    min_time = cached_df["open_time"].max()
                else:
                    cached_df = None
            except Exception as e:
                # 캐시 CSV가 손상됐으면 증분 대신 안전하게 전체 재수집으로
                # 폴백한다 (오래된/깨진 캐시가 그대로 섞여 들어가는 것을 방지).
                with print_lock:
                    log.warning(f"[{coin}] 캐시 CSV 읽기 실패, 이번 회차는 전체 재수집으로 폴백: {e}")
                cached_df = None
                min_time = None

        try:
            df, is_partial = fetch_minute_history(UPBIT_MARKETS[coin], days=days, on_page=_on_page,
                                                   rate_limiter=history_rate_limiter, min_time=min_time)
            if cached_df is not None and len(cached_df):
                # 증분으로 받은 새 데이터(df, 비어 있을 수도 있음)를 캐시와
                # 합치고 중복(겹치는 경계 시점)을 제거한 뒤 90일(=days) cutoff를
                # 다시 적용한다 — "새 데이터만 쌓여서 캐시가 무한정 길어지는"
                # 일이 없도록 매번 재수집과 동일한 보관 범위를 유지한다.
                merged = pd.concat([cached_df, df], ignore_index=True) if len(df) else cached_df
                merged = merged.drop_duplicates(subset="open_time", keep="last").sort_values(
                    "open_time").reset_index(drop=True)
                cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
                df = merged[merged["open_time"] >= cutoff].reset_index(drop=True)
            if cache and len(df):
                df.to_csv(cache_path, index=False)
            with print_lock:
                if min_time is not None:
                    log.info(f"[{coin}] 증분 수집 완료 (누적 {len(df)}개, "
                             f"캐시 마지막 시점 {min_time} 이후분만 신규 요청)")
                elif is_partial:
                    # 부분 성공(80% 이상 모은 뒤 페이지 실패, 또는 max_pages 도달)을
                    # "완료"와 구분해서 남긴다 — 이 경우 아래에서 force_full 마커를
                    # 기록하지 않아, 다음 재학습 슬롯에서 전체 재수집을 다시
                    # 시도한다 (analysis_team execution 리포트
                    # 2026-11-22_1418_execution.md 발견점 #1, 구체적 제안 #1).
                    log.warning(f"[{coin}] {len(df)}개 1분봉 부분 재수집 (목표 {days}일 중 "
                                f"일부만 수집됨) — 완료로 기록하지 않고 다음 슬롯에서 재시도")
                else:
                    log.info(f"[{coin}] {len(df)}개 1분봉 전체 재수집 완료")
            # force_full(=이번에 전체 재수집을 "시도"했는지)와 별개로, 그 시도가
            # 부분 성공(is_partial)이었으면 "오늘 완료" 마커를 기록하지 않는다 —
            # 그래야 다음 재학습 슬롯에서도 같은 날 전체 재수집을 다시 시도해,
            # 학습 데이터 창이 줄어든 채로 당일 나머지 슬롯 동안 고착되지 않는다.
            return coin, df, (force_full and not is_partial)
        except Exception as e:
            # 위 _fetch_one과 같은 이유로 FetchError만이 아니라 모든 예외를 잡는다.
            # 캐시 폴백 분기는 기존과 동일하게 그대로 탄다. 반복되는 실패는
            # _log_fetch_exception()이 1시간에 1회만 전체 스택트레이스를 남긴다.
            with print_lock:
                _log_fetch_exception(f"[WARN] {coin} 히스토리 수집 실패 (캐시 확인)", coin, e)
            if cache_path.exists():
                # 캐시 CSV 폴백 read 자체가 try/except 밖에 있어서, 캐시 파일이
                # 손상되었거나 동시 쓰기 중이면 이 read마저 실패해 예외가
                # ThreadPoolExecutor의 future.result()를 통해 그대로
                # fetch_all_minute_history() 밖(결국 retrain_lstm() 호출부, 거기도
                # try/except 없음)까지 전파됐다. "코인 1개"의 이중 실패(네트워크
                # 실패 + 캐시도 손상)가 "이번 재학습 슬롯 전체(6코인 모두)"를
                # 실패시키는, 이 파일이 다른 모든 곳에서 지키는 "코인 하나의
                # 장애가 다른 코인을 막지 않는다" 원칙이 깨지는 유일한 경로였다
                # (analysis_team execution 리포트 2026-11-22_1219 발견점 #1,
                # synthesizer Top3 #1). 이 read도 자체적으로 감싸서, 실패하면
                # 이 코인만 None으로 빠지고 나머지 코인은 영향받지 않는다.
                try:
                    return coin, pd.read_csv(cache_path, parse_dates=["open_time"]), False
                except Exception as cache_e:
                    with print_lock:
                        log.warning(f"[{coin}] 캐시 CSV 폴백 read도 실패, 이 코인만 이번 "
                                    f"슬롯 건너뜀 (다른 코인에는 영향 없음): {cache_e}")
                    return coin, None, False
            return coin, None, False

    with ThreadPoolExecutor(max_workers=len(COINS)) as executor:
        futures = {
            executor.submit(_fetch_one, coin, i): coin
            for i, coin in enumerate(COINS, start=1)
        }
        for future in as_completed(futures):
            # used_force_full == True는 "이번에 전체 재수집을 시도했고, 그 결과가
            # 부분 성공이 아니었다"(_fetch_one의 `force_full and not is_partial`)
            # 는 뜻이다. 부분 성공은 여기서 False로 들어와 마커를 남기지 않는다.
            coin, df, used_force_full = future.result()
            if df is not None:
                result[coin] = df
                if used_force_full:
                    last_full_refresh[coin] = today_str

    # 스레드 풀이 모두 끝난 뒤 한 번만 기록 (위 설명대로 경합 방지).
    _save_last_full_refresh(last_full_refresh)

    # dict 삽입 순서가 코인 순서와 안 맞을 수 있으니(완료되는 대로 들어옴),
    # 이후 코드가 COINS 순서를 가정하지 않도록 COINS 순서로 재정렬해서 반환.
    return {coin: result[coin] for coin in COINS if coin in result}


def fetch_all_latest_prices() -> dict[str, float]:
    prices: dict[str, float] = {}
    for coin in COINS:
        try:
            url = f"{UPBIT_BASE}/v1/ticker?markets={UPBIT_MARKETS[coin]}"
            data = _http_get_json(url)
            prices[coin] = float(data[0]["trade_price"])
        except Exception as e:
            # fetch_all_latest_candles/fetch_all_minute_history와 동일하게,
            # FetchError(네트워크/HTTP 재시도 소진)뿐 아니라 Upbit 응답 형식이
            # 예상과 달라 발생하는 KeyError/IndexError류까지 모두 잡아서
            # 코인 1개의 오류가 fetch_all_latest_prices() 전체를 중단시키지
            # 않도록 한다. (analysis_team 리포트 2026-11-16_1521_execution.md)
            log.exception(f"[WARN] {coin} 현재가 수집 실패: {e}")
    return prices


if __name__ == "__main__":
    print("최신 1분봉 테스트 수집...")
    candles = fetch_all_latest_candles(count=5)
    for coin, df in candles.items():
        print(f"  {coin}: {len(df)}개, 마지막 종가={df['close'].iloc[-1] if len(df) else None}")
