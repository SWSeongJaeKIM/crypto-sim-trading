"""
전략 파라미터 자동 학습/재조정 모듈.

방식 (설명 가능한 "적응형 탐색", RL이 아님):
1. 각 전략의 최근 성과(호출하는 쪽이 DB에서 읽어 전달한 snapshot 리스트)를 확인한다.
2. 최근 성과가 기준치보다 나쁘면, 그 전략의 파라미터를 미리 정의된
   탐색 범위 안에서 무작위로 소폭 변경한 "변이(variant)"를 만든다.
3. 변이는 과거 캔들 데이터로 즉시 백테스트해서, 원래 파라미터보다
   성과가 좋을 때만 교체한다 (성과가 나쁘면 원래 파라미터 유지).
4. 조정 내역(왜 바꿨는지, 전후 성과)을 반환값에 담아 설명 가능성을 유지한다.

중요 (아키텍처): 이 모듈은 저장소를 모르는 순수 함수만 제공한다.
현재 파라미터와 최근 스냅샷은 함수 인자로 받고, 결과(새 파라미터,
조정 로그)는 반환값으로 돌려준다. 실제 DB 읽기/쓰기(db_get/db_write)는
스케줄 세션의 Claude가 engine.run_cycle() 호출 전후에 직접 수행한다.

이 방식은 강화학습처럼 "스스로 경험에서 배우는" 흐름을 갖지만,
각 결정이 사람이 읽을 수 있는 백테스트 숫자로 정당화되기 때문에
블랙박스가 아니다.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone

import pandas as pd

from strategies import ALL_STRATEGIES, Strategy
from portfolio import Portfolio

# 각 전략 파라미터별 탐색 범위 (min, max, step) — 이 안에서만 변이 생성
SEARCH_SPACE: dict[str, dict[str, tuple[float, float, float]]] = {
    "ma_cross_rsi": {
        "short_window": (5, 15, 1),
        "long_window": (20, 40, 2),
        "rsi_buy_max": (55, 75, 5),
        "rsi_sell_min": (25, 45, 5),
    },
    "bollinger_rsi": {
        "bb_period": (14, 30, 2),
        "bb_std": (1.5, 2.5, 0.25),
        "rsi_oversold": (20, 35, 5),
        "rsi_overbought": (65, 80, 5),
    },
    "macd_momentum": {
        "fast": (8, 16, 2),
        "slow": (20, 32, 2),
        "signal": (6, 12, 2),
    },
    "trend_follow": {
        "ma_window": (30, 80, 5),
        "buffer_pct": (0.2, 1.0, 0.1),
    },
}

PERFORMANCE_LOOKBACK = 20   # 최근 N개 스냅샷을 "최근 성과"로 본다
UNDERPERFORM_THRESHOLD = 0.0  # 최근 수익률이 이 값보다 낮으면 재조정 검토


def recent_performance(snapshots: list[dict], lookback: int = PERFORMANCE_LOOKBACK) -> float | None:
    """snapshots: [{"timestamp":..., "pnl_pct":...}, ...] 시간순 정렬된 리스트
    (호출하는 쪽이 DB의 snapshots 컬렉션에서 해당 전략의 최근 N개를 읽어 전달).
    최근 lookback개 구간의 수익률 변화(%p)를 반환. 데이터 부족하면 None."""
    if len(snapshots) < 2:
        return None
    recent = snapshots[-lookback:] if len(snapshots) >= lookback else snapshots
    return recent[-1]["pnl_pct"] - recent[0]["pnl_pct"]


def _random_variant(strategy_name: str, base_params: dict) -> dict:
    space = SEARCH_SPACE.get(strategy_name, {})
    variant = dict(base_params)
    for key, (lo, hi, step) in space.items():
        if key not in variant:
            continue
        n_steps = max(1, round((hi - lo) / step))
        new_val = lo + step * random.randint(0, n_steps)
        if isinstance(variant[key], int):
            new_val = int(round(new_val))
        variant[key] = new_val
    return variant


def backtest(strategy_cls: type[Strategy], params: dict, candles: dict[str, pd.DataFrame],
             fx_krw_per_usdt: float = 1450.0) -> float:
    """주어진 파라미터로 전체 히스토리 캔들에 대해 단순 백테스트, 최종 수익률(%) 반환.
    각 코인은 자신이 가진 캔들 전체(상장일부터)를 쓴다 — 코인마다 길이가 달라도 됨.

    신호는 strategy.signal_series(df)로 코인당 한 번에 벡터 계산한다 (지표를
    매 캔들마다 처음부터 다시 계산하는 signal() 반복 호출보다 훨씬 빠름).
    매매 자체는 상태(현금/보유량)가 있어 캔들 순서대로 순회해야 하지만,
    신호 계산과 분리했으므로 그 루프는 가벼운 인덱싱만 한다."""
    strat = strategy_cls(**params)
    pf = Portfolio.new(f"_backtest_{strategy_cls.name}", coins=list(candles.keys()))
    last_prices = {}
    for coin, df in candles.items():
        if df.empty:
            continue
        min_bars = strat.min_bars()
        if len(df) <= min_bars:
            continue
        signals = strat.signal_series(df)
        closes = df["close"].to_numpy()
        sig_values = signals.to_numpy()
        for i in range(min_bars, len(df)):
            sig = sig_values[i]
            if sig == "HOLD":
                continue
            price = float(closes[i]) * fx_krw_per_usdt
            if sig == "BUY":
                pf.buy(coin, price, pct_of_cash=0.5, reason="backtest")
            elif sig == "SELL":
                pf.sell(coin, price, pct_of_qty=0.5, reason="backtest")
        if len(df) > 0:
            last_prices[coin] = float(closes[-1]) * fx_krw_per_usdt
    return pf.pnl_pct(last_prices)


def tune_strategy(strategy_name: str, current_params: dict, candles: dict[str, pd.DataFrame],
                   snapshots: list[dict] | None = None, n_trials: int = 8) -> dict:
    """
    한 전략에 대해 현재 성과를 확인하고, 저조하면 변이 탐색 후 개선되면 교체.

    인자:
      current_params: 호출하는 쪽이 DB의 params/<strategy_name> 문서에서 읽어온 현재 파라미터
                       (문서가 없으면 strategies.ALL_STRATEGIES[name].default_params() 사용)
      snapshots: DB의 snapshots 컬렉션에서 읽어온 해당 전략의 시간순 스냅샷 리스트 (없으면 None)

    반환: {"changed": bool, "old_params":..., "new_params":..., "old_score":..., "new_score":..., "reason":...}
    호출하는 쪽은 changed=True일 때만 params/<strategy_name> 문서를 db_write로 갱신하면 된다.
    """
    strategy_cls = ALL_STRATEGIES[strategy_name]
    snapshots = snapshots or []

    recent_perf = recent_performance(snapshots)
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "strategy": strategy_name,
        "recent_perf_pp": recent_perf,
        "changed": False,
        "old_params": current_params,
        "new_params": current_params,
    }

    # 최근 성과가 기준 이상이면 건드리지 않음 (데이터 부족(None)일 때는 탐색 진행)
    if recent_perf is not None and recent_perf >= UNDERPERFORM_THRESHOLD:
        result["reason"] = "성과가 기준 이상이라 유지"
        return result

    baseline_score = backtest(strategy_cls, current_params, candles)
    best_params, best_score = current_params, baseline_score

    for _ in range(n_trials):
        variant = _random_variant(strategy_name, current_params)
        score = backtest(strategy_cls, variant, candles)
        if score > best_score:
            best_params, best_score = variant, score

    if best_params != current_params:
        result.update({
            "changed": True,
            "new_params": best_params,
            "old_score": baseline_score,
            "new_score": best_score,
            "reason": f"백테스트 수익률 {baseline_score:.2f}% -> {best_score:.2f}% 개선",
        })
    else:
        result["old_score"] = baseline_score
        result["new_score"] = baseline_score
        result["reason"] = f"{n_trials}회 탐색했지만 기존 파라미터({baseline_score:.2f}%)보다 나은 변이를 못 찾음"

    return result


def tune_all(current_params_by_strategy: dict[str, dict], candles: dict[str, pd.DataFrame],
             snapshots_by_strategy: dict[str, list[dict]] | None = None,
             n_trials: int = 8) -> list[dict]:
    """모든 전략에 대해 tune_strategy를 실행. 각 dict는 전략이름 -> 값."""
    snapshots_by_strategy = snapshots_by_strategy or {}
    results = []
    for name in ALL_STRATEGIES:
        params = current_params_by_strategy.get(name) or ALL_STRATEGIES[name].default_params()
        snaps = snapshots_by_strategy.get(name, [])
        results.append(tune_strategy(name, params, candles, snapshots=snaps, n_trials=n_trials))
    return results


if __name__ == "__main__":
    # 네트워크 없이 모듈 자체 동작 확인용 합성 데이터 테스트
    import numpy as np

    np.random.seed(0)
    n = 300
    candles = {}
    for coin in ["BTC", "ETH"]:
        t = pd.date_range("2026-01-01", periods=n, freq="h")
        price = 100 + np.cumsum(np.random.randn(n) * 2)
        price = np.clip(price, 1, None)
        candles[coin] = pd.DataFrame({
            "open_time": t, "open": price, "high": price * 1.001,
            "low": price * 0.999, "close": price, "volume": 1,
        })

    for name in ALL_STRATEGIES:
        default_params = ALL_STRATEGIES[name].default_params()
        r = tune_strategy(name, default_params, candles, snapshots=[], n_trials=5)
        print(name, "->", r["reason"])
