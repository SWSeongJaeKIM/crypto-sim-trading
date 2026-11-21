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
                json.JSONDecodeError, UnicodeDecodeError) as e:
            # JSONDecodeError/UnicodeDecodeError: Upbit이 손상되거나 빈 응답,
            # 혹은 HTML 에러 페이지를 반환하는 경우가 드물게 있는데, 예전에는
            # 이 두 예외가 여기서 안 잡혀서 재시도 없이 바로 호출부까지
            # 전파되며 프로그램이 죽을 수 있었다. 다른 네트워크 에러와 동일하게
            # 재시도 대상으로 묶는다.
            last_err = e
            time.sleep(1.0 * (attempt + 1))
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


def fetch_latest_minute_candles(market: str, count: int = 200) -> pd.DataFrame:
    """가장 최신 1분봉 count개 (실시간 신호 계산용). 마지막 행이 가장 최근 캔들."""
    url = f"{UPBIT_BASE}/v1/candles/minutes/1?market={market}&count={count}"
    raw = _http_get_json(url)
    if not raw:
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])
    return _parse_minute_df(raw)


def fetch_minute_history(market: str, days: int = 14, sleep_sec: float = 0.12,
                          max_pages: int | None = None, on_page=None,
                          rate_limiter: "_RateLimiter | None" = None) -> pd.DataFrame:
    """최근 `days`일치 1분봉 전체를 페이지네이션으로 수집 (LSTM 학습용).
    Upbit은 페이지당 최대 200개(=200분)만 주므로, days=14면 약 14*1440/200≈101회 호출.
    `to` 파라미터로 과거 방향 페이지네이션.

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

        if oldest_time <= cutoff or len(raw) < 200:
            break
        if rate_limiter is None:
            time.sleep(sleep_sec)
    else:
        # 위와 동일한 이유로 log.warning 사용 (run.log에 남기기 위함).
        log.warning(f"{market} 수집이 max_pages({max_pages})에 도달해 중단됨 "
                    f"(목표 {days}일 중 일부만 수집됐을 수 있음)")

    if not all_rows:
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])

    df = _parse_minute_df(all_rows)
    return df[df["open_time"] >= cutoff].reset_index(drop=True)


def fetch_all_latest_candles(count: int = 200) -> dict[str, pd.DataFrame]:
    """6대 코인의 최신 1분봉 (매분 실행되는 메인 루프에서 호출). 코인당 호출이
    1번뿐이라 원래도 느리진 않지만, 순차 대신 동시에 보내서 사이클 자체를
    조금이라도 더 짧게 만든다."""
    result: dict[str, pd.DataFrame] = {}

    def _fetch_one(coin: str) -> tuple[str, pd.DataFrame | None]:
        try:
            return coin, fetch_latest_minute_candles(UPBIT_MARKETS[coin], count=count)
        except Exception as e:
            # FetchError(네트워크/HTTP 재시도 소진)뿐 아니라, Upbit 응답 형식이
            # 예상과 달라 _parse_minute_df 등에서 터지는 KeyError류까지 전부 잡아서
            # 코인 1개의 오류가 ThreadPoolExecutor를 통해 전파되어 6코인 전체
            # 수집(fetch_all_latest_candles)을 실패시키지 않도록 한다. 스택트레이스는
            # logging.exception으로 남겨 원인 진단이 가능하게 한다.
            log.exception(f"[WARN] {coin} 최신 캔들 수집 실패: {e}")
            return coin, None

    with ThreadPoolExecutor(max_workers=len(COINS)) as executor:
        for future in as_completed(executor.submit(_fetch_one, coin) for coin in COINS):
            coin, df = future.result()
            if df is not None:
                result[coin] = df
    return result


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

    def _fetch_one(coin: str, i: int) -> tuple[str, pd.DataFrame | None]:
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
        try:
            df = fetch_minute_history(UPBIT_MARKETS[coin], days=days, on_page=_on_page,
                                       rate_limiter=history_rate_limiter)
            if cache and len(df):
                df.to_csv(cache_path, index=False)
            with print_lock:
                log.info(f"[{coin}] {len(df)}개 1분봉 수집 완료")
            return coin, df
        except Exception as e:
            # 위 _fetch_one과 같은 이유로 FetchError만이 아니라 모든 예외를 잡는다.
            # 캐시 폴백 분기는 기존과 동일하게 그대로 탄다.
            with print_lock:
                log.exception(f"[WARN] {coin} 히스토리 수집 실패: {e}, 캐시 확인")
            if cache_path.exists():
                return coin, pd.read_csv(cache_path, parse_dates=["open_time"])
            return coin, None

    with ThreadPoolExecutor(max_workers=len(COINS)) as executor:
        futures = {
            executor.submit(_fetch_one, coin, i): coin
            for i, coin in enumerate(COINS, start=1)
        }
        for future in as_completed(futures):
            coin, df = future.result()
            if df is not None:
                result[coin] = df

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
