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
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
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


def fetch_minute_history(market: str, days: int = 14, sleep_sec: float = 0.12, max_pages: int = 1000) -> pd.DataFrame:
    """최근 `days`일치 1분봉 전체를 페이지네이션으로 수집 (LSTM 학습용).
    Upbit은 페이지당 최대 200개(=200분)만 주므로, days=14면 약 14*1440/200≈101회 호출.
    `to` 파라미터로 과거 방향 페이지네이션."""
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    all_rows: list[dict] = []
    to_param = ""

    for _ in range(max_pages):
        url = f"{UPBIT_BASE}/v1/candles/minutes/1?market={market}&count=200"
        if to_param:
            url += f"&to={to_param}"
        raw = _http_get_json(url)
        if not raw:
            break
        all_rows.extend(raw)
        oldest = raw[-1]
        oldest_time = pd.to_datetime(oldest["candle_date_time_kst"])
        to_param = oldest["candle_date_time_utc"]
        if oldest_time <= cutoff or len(raw) < 200:
            break
        time.sleep(sleep_sec)

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
    result: dict[str, pd.DataFrame] = {}
    for i, coin in enumerate(COINS, start=1):
        if on_progress:
            on_progress(coin, i, len(COINS))
        cache_path = DATA_DIR / f"{coin}_1m_{days}d.csv"
        try:
            df = fetch_minute_history(UPBIT_MARKETS[coin], days=days)
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
