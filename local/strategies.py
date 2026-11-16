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
    # avg_loss == 0인 구간은 rs가 NaN이 되어 fillna(50.0)으로 "중립"이 되어버리지만,
    # 실제로는 avg_gain > 0이면 극단적 과매수(RSI=100), avg_gain도 0이면 변동 없음(RSI=50)이다.
    # (analysis_team 리포트 2026-11-16_1521_strategy.md 참고)
    out = out.where(avg_loss != 0, 100.0)
    out = out.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
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

    def signal_with_confidence(self, df: pd.DataFrame) -> tuple[Signal, float]:
        """신호와 함께 "이번 신호를 얼마나 확신하는지"(0.0~1.0)를 반환한다.
        main_loop.py가 이 값으로 베팅 크기(자금의 몇 %를 움직일지)를 정한다
        — 확신도가 낮으면 적게, 높으면 많이 베팅. 기본 구현은 모든 신호를
        동일하게 확신(1.0)하는 것으로 처리해, 오버라이드하지 않은 전략도
        기존과 동일하게 동작한다(하위호환). 신호 강도를 계산할 수 있는
        전략은 이 메서드를 오버라이드해서 더 정교한 값을 준다."""
        return self.signal(df), 1.0

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

    def signal_with_confidence(self, df: pd.DataFrame) -> tuple[Signal, float]:
        """RSI가 임계값에서 얼마나 여유 있게 떨어져 있는지로 확신도를 매긴다.
        예: BUY인데 RSI가 임계값(65) 바로 아래(64)면 애매한 크로스라 확신도↓,
        RSI가 30처럼 임계값과 멀리 떨어져 있으면 더 뚜렷한 상황이라 확신도↑."""
        sig = self.signal(df)
        if sig == "HOLD" or len(df) < self.min_bars():
            return sig, 0.0
        p = self.params
        r = rsi(df["close"], p["rsi_period"])
        curr_rsi = r.iloc[-1]
        if sig == "BUY":
            # RSI가 0에 가까울수록(과매도) 더 확신, rsi_buy_max에 가까울수록 애매
            margin = (p["rsi_buy_max"] - curr_rsi) / p["rsi_buy_max"]
        else:  # SELL
            margin = (curr_rsi - p["rsi_sell_min"]) / (100 - p["rsi_sell_min"])
        return sig, float(np.clip(margin, 0.0, 1.0))


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

    def signal_with_confidence(self, df: pd.DataFrame) -> tuple[Signal, float]:
        """가격이 밴드를 얼마나 많이 뚫고 나갔는지(밴드폭 대비 %)로 확신도를 매긴다.
        살짝 터치만 한 경우보다 많이 뚫고 나간 경우가 더 뚜렷한 평균회귀 신호."""
        sig = self.signal(df)
        if sig == "HOLD" or len(df) < self.min_bars():
            return sig, 0.0
        p = self.params
        close = df["close"]
        upper, mid, lower = bollinger(close, p["bb_period"], p["bb_std"])
        band_width = max(upper.iloc[-1] - lower.iloc[-1], 1e-9)
        price = close.iloc[-1]
        if sig == "BUY":
            overshoot = (lower.iloc[-1] - price) / band_width
        else:  # SELL
            overshoot = (price - upper.iloc[-1]) / band_width
        # 밴드폭의 30%만큼 더 뚫고 나가면 "확실한" 신호로 간주(그 이상은 1.0으로 캡)
        confidence = overshoot / 0.3
        return sig, float(np.clip(confidence, 0.0, 1.0))


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

    def signal_with_confidence(self, df: pd.DataFrame) -> tuple[Signal, float]:
        """크로스 직후 히스토그램이 얼마나 빠르게 벌어지는지(기울기)로 확신도를
        매긴다. 살짝 양수로 넘어간 것보다 확실하게 치고 올라간 경우가 더
        뚜렷한 모멘텀 전환 신호."""
        sig = self.signal(df)
        if sig == "HOLD" or len(df) < self.min_bars():
            return sig, 0.0
        p = self.params
        close = df["close"]
        _, _, hist = macd(close, p["fast"], p["slow"], p["signal"])
        curr_hist = hist.iloc[-1]
        # 최근 20개 캔들의 히스토그램 표준편차로 정규화 — 평소 변동폭 대비
        # 이번 전환이 얼마나 큰지를 본다(코인/기간마다 스케일이 다르므로).
        recent_std = hist.iloc[-20:].std()
        if not recent_std or np.isnan(recent_std) or recent_std < 1e-9:
            return sig, 0.5  # 변동성 추정이 불안정하면 중간값으로
        confidence = abs(curr_hist) / (recent_std * 1.5)
        return sig, float(np.clip(confidence, 0.0, 1.0))


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

    def signal_with_confidence(self, df: pd.DataFrame) -> tuple[Signal, float]:
        """가격이 이동평균 밴드를 얼마나 멀리 벗어났는지(이동평균 대비 %)로
        확신도를 매긴다. 밴드를 살짝 넘은 경우보다 뚜렷하게 추세가 붙은
        경우가 더 강한 신호."""
        sig = self.signal(df)
        if sig == "HOLD" or len(df) < self.min_bars():
            return sig, 0.0
        p = self.params
        close = df["close"]
        ma_now = sma(close, p["ma_window"]).iloc[-1]
        price = close.iloc[-1]
        deviation_pct = abs(price - ma_now) / ma_now * 100
        # buffer_pct(기본 0.5%)의 4배(2%)만큼 벗어나면 "확실한" 추세로 간주
        confidence = deviation_pct / (p["buffer_pct"] * 4)
        return sig, float(np.clip(confidence, 0.0, 1.0))


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
