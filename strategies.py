"""
조건식(매매 전략) 라이브러리.

모든 전략은 OHLCV DataFrame(시간순 정렬, 컬럼: open,high,low,close,volume)을 받아서
마지막 시점 기준 신호를 돌려준다: "BUY", "SELL", "HOLD".

전략은 파라미터를 dict로 가지며, learner.py가 이 파라미터를 성과 기반으로
조정할 수 있도록 get_params()/set_params()를 지원한다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

Signal = Literal["BUY", "SELL", "HOLD"]


# ---------- 지표 계산 헬퍼 ----------

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(series: pd.Series, period: int = 20, num_std: float = 2.0):
    mid = sma(series, period)
    std = series.rolling(window=period, min_periods=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


# ---------- 전략 베이스 클래스 ----------

class Strategy(ABC):
    name: str = "base"

    def __init__(self, **params):
        self.params = dict(self.default_params())
        self.params.update(params)

    @classmethod
    @abstractmethod
    def default_params(cls) -> dict:
        ...

    @abstractmethod
    def signal(self, df: pd.DataFrame) -> Signal:
        ...

    def signal_series(self, df: pd.DataFrame) -> pd.Series:
        """df 전체 길이에 대한 신호를 한 번에 계산해 Series(값: BUY/SELL/HOLD)로 반환.
        백테스트(learner.backtest)에서 쓰는 벡터화 경로 — 지표를 한 번만 계산하고
        매 캔들마다 df.iloc[:i+1]로 다시 계산하는 것을 피해 속도를 크게 높인다.
        각 서브클래스가 오버라이드하지 않으면, 느리지만 항상 맞는 fallback으로
        signal()을 반복 호출한다."""
        out = []
        min_bars = self.min_bars()
        for i in range(len(df)):
            if i + 1 < min_bars:
                out.append("HOLD")
            else:
                out.append(self.signal(df.iloc[: i + 1]))
        return pd.Series(out, index=df.index)

    def get_params(self) -> dict:
        return dict(self.params)

    def set_params(self, **kwargs):
        self.params.update(kwargs)

    def min_bars(self) -> int:
        """신호 계산에 필요한 최소 캔들 수. 부족하면 HOLD 처리용."""
        return 50


# ---------- 전략 1: 이동평균 골든/데드크로스 + RSI 필터 ----------

class MACrossRSIStrategy(Strategy):
    name = "ma_cross_rsi"

    @classmethod
    def default_params(cls) -> dict:
        return {
            "short_window": 9,
            "long_window": 26,
            "rsi_period": 14,
            "rsi_buy_max": 65,   # RSI가 이 값보다 낮을 때만 매수 허용(과매수 방지)
            "rsi_sell_min": 35,  # RSI가 이 값보다 높을 때만 매도 허용(과매도 패닉셀 방지)
        }

    def min_bars(self) -> int:
        return max(self.params["long_window"], self.params["rsi_period"]) + 2

    def signal(self, df: pd.DataFrame) -> Signal:
        p = self.params
        if len(df) < self.min_bars():
            return "HOLD"
        close = df["close"]
        short_ma = sma(close, p["short_window"])
        long_ma = sma(close, p["long_window"])
        r = rsi(close, p["rsi_period"])

        prev_diff = short_ma.iloc[-2] - long_ma.iloc[-2]
        curr_diff = short_ma.iloc[-1] - long_ma.iloc[-1]
        curr_rsi = r.iloc[-1]

        golden_cross = prev_diff <= 0 and curr_diff > 0
        dead_cross = prev_diff >= 0 and curr_diff < 0

        if golden_cross and curr_rsi < p["rsi_buy_max"]:
            return "BUY"
        if dead_cross and curr_rsi > p["rsi_sell_min"]:
            return "SELL"
        return "HOLD"

    def signal_series(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        close = df["close"]
        short_ma = sma(close, p["short_window"])
        long_ma = sma(close, p["long_window"])
        r = rsi(close, p["rsi_period"])
        diff = short_ma - long_ma
        prev_diff = diff.shift(1)

        golden_cross = (prev_diff <= 0) & (diff > 0)
        dead_cross = (prev_diff >= 0) & (diff < 0)
        buy = golden_cross & (r < p["rsi_buy_max"])
        sell = dead_cross & (r > p["rsi_sell_min"])

        out = pd.Series("HOLD", index=df.index)
        out[buy] = "BUY"
        out[sell] = "SELL"
        min_bars = self.min_bars()
        out.iloc[: min_bars - 1] = "HOLD"
        return out


# ---------- 전략 2: 볼린저밴드 + RSI (평균회귀) ----------

class BollingerRSIStrategy(Strategy):
    name = "bollinger_rsi"

    @classmethod
    def default_params(cls) -> dict:
        return {
            "bb_period": 20,
            "bb_std": 2.0,
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
        }

    def min_bars(self) -> int:
        return max(self.params["bb_period"], self.params["rsi_period"]) + 2

    def signal(self, df: pd.DataFrame) -> Signal:
        p = self.params
        if len(df) < self.min_bars():
            return "HOLD"
        close = df["close"]
        upper, mid, lower = bollinger(close, p["bb_period"], p["bb_std"])
        r = rsi(close, p["rsi_period"])

        price = close.iloc[-1]
        curr_rsi = r.iloc[-1]

        # 하단 밴드 터치 + RSI 과매도 => 반등 기대 매수
        if price <= lower.iloc[-1] and curr_rsi < p["rsi_oversold"]:
            return "BUY"
        # 상단 밴드 터치 + RSI 과매수 => 되돌림 기대 매도
        if price >= upper.iloc[-1] and curr_rsi > p["rsi_overbought"]:
            return "SELL"
        return "HOLD"

    def signal_series(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        close = df["close"]
        upper, mid, lower = bollinger(close, p["bb_period"], p["bb_std"])
        r = rsi(close, p["rsi_period"])

        buy = (close <= lower) & (r < p["rsi_oversold"])
        sell = (close >= upper) & (r > p["rsi_overbought"])

        out = pd.Series("HOLD", index=df.index)
        out[buy] = "BUY"
        out[sell] = "SELL"
        min_bars = self.min_bars()
        out.iloc[: min_bars - 1] = "HOLD"
        return out


# ---------- 전략 3: MACD 모멘텀 ----------

class MACDStrategy(Strategy):
    name = "macd_momentum"

    @classmethod
    def default_params(cls) -> dict:
        return {
            "fast": 12,
            "slow": 26,
            "signal": 9,
        }

    def min_bars(self) -> int:
        return self.params["slow"] + self.params["signal"] + 2

    def signal(self, df: pd.DataFrame) -> Signal:
        p = self.params
        if len(df) < self.min_bars():
            return "HOLD"
        close = df["close"]
        macd_line, signal_line, hist = macd(close, p["fast"], p["slow"], p["signal"])

        prev_hist = hist.iloc[-2]
        curr_hist = hist.iloc[-1]

        # MACD 히스토그램이 음에서 양으로 전환 => 매수
        if prev_hist <= 0 and curr_hist > 0:
            return "BUY"
        # 양에서 음으로 전환 => 매도
        if prev_hist >= 0 and curr_hist < 0:
            return "SELL"
        return "HOLD"

    def signal_series(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        close = df["close"]
        _, _, hist = macd(close, p["fast"], p["slow"], p["signal"])
        prev_hist = hist.shift(1)

        buy = (prev_hist <= 0) & (hist > 0)
        sell = (prev_hist >= 0) & (hist < 0)

        out = pd.Series("HOLD", index=df.index)
        out[buy] = "BUY"
        out[sell] = "SELL"
        min_bars = self.min_bars()
        out.iloc[: min_bars - 1] = "HOLD"
        return out


# ---------- 전략 4: 단순 추세추종 (장기 이동평균 위/아래) ----------

class TrendFollowStrategy(Strategy):
    name = "trend_follow"

    @classmethod
    def default_params(cls) -> dict:
        return {
            "ma_window": 50,
            "buffer_pct": 0.5,  # MA 대비 ±0.5% 밴드 안에서는 HOLD(노이즈 방지)
        }

    def min_bars(self) -> int:
        return self.params["ma_window"] + 2

    def signal(self, df: pd.DataFrame) -> Signal:
        p = self.params
        if len(df) < self.min_bars():
            return "HOLD"
        close = df["close"]
        ma = sma(close, p["ma_window"])
        price = close.iloc[-1]
        prev_price = close.iloc[-2]
        ma_now = ma.iloc[-1]
        ma_prev = ma.iloc[-2]

        band = ma_now * p["buffer_pct"] / 100

        crossed_up = prev_price <= (ma_prev + band) and price > (ma_now + band)
        crossed_down = prev_price >= (ma_prev - band) and price < (ma_now - band)

        if crossed_up:
            return "BUY"
        if crossed_down:
            return "SELL"
        return "HOLD"

    def signal_series(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        close = df["close"]
        ma = sma(close, p["ma_window"])
        prev_price = close.shift(1)
        prev_ma = ma.shift(1)
        band = ma * p["buffer_pct"] / 100

        crossed_up = (prev_price <= (prev_ma + band)) & (close > (ma + band))
        crossed_down = (prev_price >= (prev_ma - band)) & (close < (ma - band))

        out = pd.Series("HOLD", index=df.index)
        out[crossed_up] = "BUY"
        out[crossed_down] = "SELL"
        min_bars = self.min_bars()
        out.iloc[: min_bars - 1] = "HOLD"
        return out


# 등록된 모든 전략 (경쟁에 참여하는 기본 셋)
ALL_STRATEGIES: dict[str, type[Strategy]] = {
    MACrossRSIStrategy.name: MACrossRSIStrategy,
    BollingerRSIStrategy.name: BollingerRSIStrategy,
    MACDStrategy.name: MACDStrategy,
    TrendFollowStrategy.name: TrendFollowStrategy,
}


def build_default_roster() -> dict[str, Strategy]:
    """전략 이름 -> 인스턴스(기본 파라미터) 매핑."""
    return {name: cls() for name, cls in ALL_STRATEGIES.items()}
