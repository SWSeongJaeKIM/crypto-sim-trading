"""
매매 엔진: 전략 신호 계산 -> 모의 매매 실행 (순수 함수, DB I/O 없음)

아키텍처:
이 모듈은 Artifact DB를 직접 호출하지 않는다. 스케줄 세션(또는 메인 세션)의
Claude가 다음 순서로 동작한다:

  1. (DB 읽기, MCP 도구) 각 전략의 현재 파라미터(params/<name>)와
     포트폴리오 상태(portfolios/<name>), 최근 스냅샷(snapshots 컬렉션)을
     db_get/db_query로 읽어온다.
  2. (데이터 수집) data_fetcher.fetch_all_candles()로 최신 캔들을 받아온다.
  3. (순수 계산) 이 모듈의 run_cycle(...)에 1,2의 결과를 넘겨서
     학습(파라미터 재조정) + 매매 신호 계산 + 모의 매수/매도를 전부
     메모리에서 수행하고, "갱신된 상태"를 반환받는다.
  4. (DB 쓰기, MCP 도구) 반환된 포트폴리오 문서, 신규 거래, 신규 스냅샷,
     갱신된 파라미터, 튜닝 로그를 db_write로 저장한다.

run_cycle()은 테스트하기 쉽도록 네트워크/DB에 전혀 의존하지 않는다.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from strategies import ALL_STRATEGIES
from portfolio import Portfolio, COINS, usdt_to_krw, INITIAL_CAPITAL_KRW
from learner import tune_strategy

TRADE_FRACTION = 0.5  # 매수/매도 시 자금의 몇 %를 움직일지


def latest_prices_krw(candles: dict[str, pd.DataFrame], source: str,
                       fx_krw_per_usdt: float = 1450.0) -> dict[str, float]:
    prices = {}
    for coin, df in candles.items():
        if df.empty:
            continue
        last_close = float(df["close"].iloc[-1])
        prices[coin] = last_close if source == "upbit" else usdt_to_krw(last_close, fx_krw_per_usdt)
    return prices


def run_cycle(
    candles: dict[str, pd.DataFrame],
    source: str,
    portfolio_docs: dict[str, dict | None],
    param_docs: dict[str, dict | None],
    snapshots_by_strategy: dict[str, list[dict]] | None = None,
    tune: bool = True,
    tune_trials: int = 8,
    fx_krw_per_usdt: float = 1450.0,
) -> dict:
    """
    한 번의 매매 사이클을 순수하게 계산한다 (DB 입출력 없음).

    인자:
      candles: {coin: OHLCV DataFrame} — data_fetcher로 받아온 최신 캔들
      source: "binance" | "upbit"
      portfolio_docs: {strategy_name: db_get한 portfolios/<name> 문서 또는 None(신규)}
      param_docs: {strategy_name: db_get한 params/<name> 문서 또는 None(기본값 사용)}
      snapshots_by_strategy: {strategy_name: [{"timestamp":..,"pnl_pct":..}, ...]} (최근 N개, 시간순)
      tune: True면 전략별 파라미터 재조정(learner.tune_strategy)을 먼저 수행

    반환 (모두 DB에 그대로 쓸 수 있는 JSON 호환 dict/list):
      {
        "timestamp": ISO8601,
        "source": "binance" | "upbit",
        "prices_krw": {coin: price},
        "tuning_results": [learner.tune_strategy 반환값, ...],
        "portfolio_docs": {strategy_name: Portfolio.to_doc()},   # db_write로 set/update
        "new_trades": {strategy_name: [trade dict, ...]},        # trades 컬렉션에 추가
        "new_snapshots": {strategy_name: {"timestamp":..,"pnl_pct":..,"total_value_krw":..}},
        "updated_params": {strategy_name: params dict},          # changed=True인 것만 포함
        "actions": [{"strategy":..,"coin":..,"signal":..,"price_krw":..}, ...],
      }
    """
    snapshots_by_strategy = snapshots_by_strategy or {}
    now = datetime.now(timezone.utc).isoformat()
    prices_krw = latest_prices_krw(candles, source, fx_krw_per_usdt)

    result: dict = {
        "timestamp": now,
        "source": source,
        "prices_krw": prices_krw,
        "tuning_results": [],
        "portfolio_docs": {},
        "new_trades": {},
        "new_snapshots": {},
        "updated_params": {},
        "actions": [],
    }

    for strat_name, strat_cls in ALL_STRATEGIES.items():
        current_params = param_docs.get(strat_name) or strat_cls.default_params()

        # 1. 학습: 최근 성과가 저조하면 백테스트 기반으로 파라미터 재조정
        if tune:
            tuning = tune_strategy(
                strat_name, current_params, candles,
                snapshots=snapshots_by_strategy.get(strat_name, []),
                n_trials=tune_trials,
            )
            result["tuning_results"].append(tuning)
            if tuning["changed"]:
                current_params = tuning["new_params"]
                result["updated_params"][strat_name] = current_params

        strat = strat_cls(**current_params)

        # 2. 포트폴리오 로드 (DB 문서 있으면 복원, 없으면 신규)
        pf_doc = portfolio_docs.get(strat_name)
        pf = Portfolio.from_doc(pf_doc) if pf_doc else Portfolio.new(strat_name, coins=COINS,
                                                                       initial_capital_krw=INITIAL_CAPITAL_KRW)

        # 3. 코인별 신호 계산 + 매매 실행
        for coin, df in candles.items():
            if df.empty or coin not in prices_krw or coin not in pf.slots:
                continue
            signal = strat.signal(df)
            price_krw = prices_krw[coin]
            acted = False
            if signal == "BUY":
                acted = pf.buy(coin, price_krw, pct_of_cash=TRADE_FRACTION,
                                reason=f"{strat_name} BUY signal")
            elif signal == "SELL":
                acted = pf.sell(coin, price_krw, pct_of_qty=TRADE_FRACTION,
                                 reason=f"{strat_name} SELL signal")
            if acted:
                result["actions"].append({
                    "strategy": strat_name, "coin": coin, "signal": signal,
                    "price_krw": price_krw,
                })

        # 4. 결과 수집
        result["portfolio_docs"][strat_name] = pf.to_doc()
        result["new_trades"][strat_name] = pf.new_trades_as_dicts()
        total_value = pf.total_value_krw(prices_krw)
        pnl_pct = pf.pnl_pct(prices_krw)
        result["new_snapshots"][strat_name] = {
            "timestamp": now,
            "total_value_krw": total_value,
            "pnl_pct": pnl_pct,
        }

    return result
