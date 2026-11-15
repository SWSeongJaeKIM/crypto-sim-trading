"""
Binance 공개 API로 시세(캔들) 데이터를 받아오는 모듈.

사용 전 확인사항:
- 이 스크립트가 도는 환경(세션)의 아웃바운드 네트워크 allowlist에
  api.binance.com 이 허용되어 있어야 합니다.
- claude.ai 사용 시: Settings > Capabilities (개인) 또는
  Organization settings > Capabilities (팀/엔터프라이즈)에서
  "api.binance.com" 을 허용 도메인에 추가한 뒤, 새 대화를 열어야 반영됩니다.

Binance가 지역/계정 문제로 막힐 경우를 대비해 Upbit(원화 마켓, KRW 기준)
fallback도 포함합니다. 단, 두 거래소는 기준 화폐가 다르므로(USDT vs KRW)
동일 전략 비교에는 한 거래소로 통일해서 쓰는 걸 권장합니다.
"""

from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# 6대 코인 (Binance 심볼 기준, USDT 마켓)
SYMBOLS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "XRP": "XRPUSDT",
    "SOL": "SOLUSDT",
    "DOGE": "DOGEUSDT",
    "ADA": "ADAUSDT",
}

BINANCE_BASE = "https://api.binance.com"
UPBIT_BASE = "https://api.upbit.com"

# Upbit 마켓 코드 (KRW 마켓) - fallback용
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
            req = urllib.request.Request(url, headers={"User-Agent": "crypto-sim/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise FetchError(f"GET failed after {retries} retries: {url} ({last_err})")


def fetch_binance_klines(symbol: str, interval: str = "1h", limit: int = 200) -> pd.DataFrame:
    """
    Binance 공개 캔들(kline) API.
    interval: 1m, 5m, 15m, 1h, 4h, 1d 등
    limit: 최대 1000
    문서: https://binance-docs.github.io/apidocs/spot/en/#kline-candlestick-data
    """
    url = f"{BINANCE_BASE}/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    raw = _http_get_json(url)
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(raw, columns=cols)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    return df[["open_time", "open", "high", "low", "close", "volume", "close_time"]]


def fetch_binance_price(symbol: str) -> float:
    """현재가(최신 체결가) 하나만 빠르게 조회."""
    url = f"{BINANCE_BASE}/api/v3/ticker/price?symbol={symbol}"
    data = _http_get_json(url)
    return float(data["price"])


def fetch_binance_klines_full_history(
    symbol: str,
    interval: str = "1d",
    start_time_ms: int | None = None,
    end_time_ms: int | None = None,
    sleep_sec: float = 0.3,
    max_pages: int = 100,
) -> pd.DataFrame:
    """
    상장일(또는 start_time_ms)부터 end_time_ms(기본: 지금)까지의 전체 캔들을 페이지네이션으로 수집.
    Binance klines는 한 번에 최대 1000개만 주므로, 마지막 캔들의 close_time+1ms를
    다음 호출의 startTime으로 넘겨가며 반복 호출한다.

    start_time_ms를 생략하면 Binance가 그 심볼에 대해 가진 가장 이른 데이터부터
    (startTime=0으로 요청하면 거래소가 알아서 상장일 근처부터 반환) 수집한다.
    즉 코인마다 실제 상장일부터 자동으로 맞춰진다 (BTC는 더 길게, SOL/DOGE/ADA는
    상장 시점부터만 모임 — 코인마다 가진 데이터 전부 사용하는 설계).

    max_pages: 무한루프 방지용 안전장치 (1000개 * max_pages 캔들까지).
    """
    all_rows: list[list] = []
    cursor = start_time_ms if start_time_ms is not None else 0
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ]

    for _ in range(max_pages):
        url = f"{BINANCE_BASE}/api/v3/klines?symbol={symbol}&interval={interval}&limit=1000&startTime={cursor}"
        if end_time_ms is not None:
            url += f"&endTime={end_time_ms}"
        raw = _http_get_json(url)
        if not raw:
            break
        all_rows.extend(raw)
        last_close_time = raw[-1][6]  # close_time 컬럼
        if len(raw) < 1000:
            break  # 마지막 페이지
        cursor = last_close_time + 1
        time.sleep(sleep_sec)  # API rate limit 매너

    if not all_rows:
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume", "close_time"])

    df = pd.DataFrame(all_rows, columns=cols)
    df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    return df[["open_time", "open", "high", "low", "close", "volume", "close_time"]]


def fetch_all_full_history(interval: str = "1d", source: str = "upbit") -> dict[str, pd.DataFrame]:
    """6대 코인 각각이 가진 전체 역사 데이터를 상장일부터 수집 (코인마다 길이 다름).
    source="upbit": 원화(KRW) 기준, 환율 변환 불필요 (기본값 — Binance는 지역 제한으로 접속 불가).
    source="binance": USDT 기준, 환율 변환 필요 (접속 가능한 환경에서만 사용)."""
    result: dict[str, pd.DataFrame] = {}
    for coin, symbol in SYMBOLS.items():
        try:
            if source == "upbit":
                df = fetch_upbit_daily_full_history(UPBIT_MARKETS[coin])
            else:
                df = fetch_binance_klines_full_history(symbol, interval=interval)
            result[coin] = df
            df.to_csv(DATA_DIR / f"{coin}_{interval}_full_history_{source}.csv", index=False)
            print(f"[{coin}] {len(df)}개 캔들 수집 "
                  f"({df['open_time'].iloc[0]} ~ {df['open_time'].iloc[-1]})" if len(df) else f"[{coin}] 데이터 없음")
        except FetchError as e:
            print(f"[WARN] {coin} 전체 히스토리 수집 실패: {e}")
    return result


def fetch_upbit_candles(market: str, unit_minutes: int = 60, count: int = 200) -> pd.DataFrame:
    """
    Upbit 공개 분봉 API (fallback). unit_minutes: 1,3,5,15,10,30,60,240
    문서: https://docs.upbit.com/reference/분minute-캔들-조회
    """
    url = f"{UPBIT_BASE}/v1/candles/minutes/{unit_minutes}?market={market}&count={count}"
    raw = _http_get_json(url)
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
    df = df.sort_values("open_time").reset_index(drop=True)
    return df[["open_time", "open", "high", "low", "close", "volume"]]


def fetch_upbit_price(market: str) -> float:
    url = f"{UPBIT_BASE}/v1/ticker?markets={market}"
    data = _http_get_json(url)
    return float(data[0]["trade_price"])


def fetch_upbit_daily_full_history(
    market: str,
    sleep_sec: float = 0.15,
    max_pages: int = 60,
) -> pd.DataFrame:
    """
    Upbit 일봉(day candles) API로 상장일부터 전체 역사를 페이지네이션 수집.
    Upbit은 한 번에 최대 200개 캔들만 주고, `to` 파라미터(이 시각 "이전" 데이터)로
    과거 방향으로 페이지를 넘긴다. 가장 오래된 캔들의 시각을 다음 호출의 `to`로 써서
    더 이상 데이터가 없을 때(캔들 수가 200개 미만)까지 반복한다.
    문서: https://docs.upbit.com/reference/일day-캔들-조회
    원화(KRW) 기준이므로 환율 변환이 필요 없다.
    """
    all_rows: list[dict] = []
    to_param = ""  # 빈 문자열이면 "지금"부터

    for _ in range(max_pages):
        url = f"{UPBIT_BASE}/v1/candles/days?market={market}&count=200"
        if to_param:
            url += f"&to={to_param}"
        raw = _http_get_json(url)
        if not raw:
            break
        all_rows.extend(raw)
        oldest = raw[-1]  # Upbit은 최신->과거 순으로 반환
        to_param = oldest["candle_date_time_utc"]
        if len(raw) < 200:
            break  # 더 이전 데이터 없음 (상장일 도달)
        time.sleep(sleep_sec)

    if not all_rows:
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_rows)
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


def fetch_all_candles(interval: str = "1h", limit: int = 200, source: str = "upbit") -> dict[str, pd.DataFrame]:
    """6대 코인 전체의 캔들 데이터를 받아와 dict로 반환하고 로컬 캐시에 저장."""
    result: dict[str, pd.DataFrame] = {}
    for coin, symbol in SYMBOLS.items():
        try:
            if source == "binance":
                df = fetch_binance_klines(symbol, interval=interval, limit=limit)
            elif source == "upbit":
                unit = {"1h": 60, "15m": 15, "4h": 240, "1d": 1440}.get(interval, 60)
                df = fetch_upbit_candles(UPBIT_MARKETS[coin], unit_minutes=unit, count=limit)
            else:
                raise ValueError(f"unknown source: {source}")
            result[coin] = df
            df.to_csv(DATA_DIR / f"{coin}_{interval}.csv", index=False)
        except FetchError as e:
            print(f"[WARN] {coin} 데이터 수집 실패: {e}")
    return result


def fetch_all_prices(source: str = "upbit") -> dict[str, float]:
    """6대 코인 현재가 스냅샷."""
    prices: dict[str, float] = {}
    for coin, symbol in SYMBOLS.items():
        try:
            if source == "binance":
                prices[coin] = fetch_binance_price(symbol)
            else:
                prices[coin] = fetch_upbit_price(UPBIT_MARKETS[coin])
        except FetchError as e:
            print(f"[WARN] {coin} 현재가 수집 실패: {e}")
    return prices


if __name__ == "__main__":
    import sys

    src = sys.argv[1] if len(sys.argv) > 1 else "binance"
    print(f"[{src}] 6대 코인 현재가 조회 중...")
    prices = fetch_all_prices(source=src)
    for coin, p in prices.items():
        print(f"  {coin}: {p}")
    if not prices:
        print("가격을 하나도 받아오지 못했습니다. 네트워크 allowlist를 확인하세요.")
