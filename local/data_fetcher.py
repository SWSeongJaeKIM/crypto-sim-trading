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
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

import pandas as pd

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
                          max_pages: int | None = None, on_page=None) -> pd.DataFrame:
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
    (진행 로그/진행률 표시용, 선택). 멈춘 건지 그냥 느린 건지 구분하려면 필수."""
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    if max_pages is None:
        max_pages = int(days * 1440 / 200 * 1.2) + 5
    all_rows: list[dict] = []
    to_param = ""
    prev_to_param = None
    stuck_count = 0

    for page_idx in range(1, max_pages + 1):
        url = f"{UPBIT_BASE}/v1/candles/minutes/1?market={market}&count=200"
        if to_param:
            url += f"&to={to_param}"
        raw = _http_get_json(url)
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
                print(f"[WARN] {market} 페이지네이션이 {new_to_param} 시점에서 "
                      f"전진하지 않아 중단 (수집된 {len(all_rows)}개로 계속)")
                break
        else:
            stuck_count = 0
        prev_to_param = new_to_param
        to_param = new_to_param

        if oldest_time <= cutoff or len(raw) < 200:
            break
        time.sleep(sleep_sec)
    else:
        print(f"[WARN] {market} 수집이 max_pages({max_pages})에 도달해 중단됨 "
              f"(목표 {days}일 중 일부만 수집됐을 수 있음)")

    if not all_rows:
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])

    df = _parse_minute_df(all_rows)
    return df[df["open_time"] >= cutoff].reset_index(drop=True)


def fetch_all_latest_candles(count: int = 200) -> dict[str, pd.DataFrame]:
    """6대 코인의 최신 1분봉 (매분 실행되는 메인 루프에서 호출)."""
    result: dict[str, pd.DataFrame] = {}
    for coin in COINS:
        try:
            result[coin] = fetch_latest_minute_candles(UPBIT_MARKETS[coin], count=count)
        except FetchError as e:
            print(f"[WARN] {coin} 최신 캔들 수집 실패: {e}")
    return result


def fetch_all_minute_history(days: int = 14, cache: bool = True, on_progress=None) -> dict[str, pd.DataFrame]:
    """6대 코인의 최근 N일 1분봉 전체 (LSTM 학습/재학습 시 호출). 디스크 캐시 지원.
    on_progress(coin, index, total): 코인 하나 수집 시작할 때마다 호출 (대시보드 진행표시용, 선택)."""
    expected_pages = int(days * 1440 / 200)
    result: dict[str, pd.DataFrame] = {}
    for i, coin in enumerate(COINS, start=1):
        if on_progress:
            on_progress(coin, i, len(COINS))

        def _on_page(page_idx, oldest_time, rows_so_far, _coin=coin):
            # 10페이지마다(약 1~2초 간격) 한 번씩만 찍어서 로그 폭주를 막되,
            # "멈춘 건지 그냥 느린 건지"는 바로 구분할 수 있게 한다.
            if page_idx % 10 == 0 or page_idx == 1:
                print(f"  [{_coin}] 페이지 {page_idx}/~{expected_pages} 수집 중... "
                      f"(누적 {rows_so_far}개, 가장 오래된 시점 {oldest_time})")

        cache_path = DATA_DIR / f"{coin}_1m_{days}d.csv"
        try:
            df = fetch_minute_history(UPBIT_MARKETS[coin], days=days, on_page=_on_page)
            result[coin] = df
            if cache and len(df):
                df.to_csv(cache_path, index=False)
            print(f"[{coin}] {len(df)}개 1분봉 수집 완료")
        except FetchError as e:
            print(f"[WARN] {coin} 히스토리 수집 실패: {e}, 캐시 확인")
            if cache_path.exists():
                result[coin] = pd.read_csv(cache_path, parse_dates=["open_time"])
    return result


def fetch_all_latest_prices() -> dict[str, float]:
    prices: dict[str, float] = {}
    for coin in COINS:
        try:
            url = f"{UPBIT_BASE}/v1/ticker?markets={UPBIT_MARKETS[coin]}"
            data = _http_get_json(url)
            prices[coin] = float(data[0]["trade_price"])
        except FetchError as e:
            print(f"[WARN] {coin} 현재가 수집 실패: {e}")
    return prices


if __name__ == "__main__":
    print("최신 1분봉 테스트 수집...")
    candles = fetch_all_latest_candles(count=5)
    for coin, df in candles.items():
        print(f"  {coin}: {len(df)}개, 마지막 종가={df['close'].iloc[-1] if len(df) else None}")
