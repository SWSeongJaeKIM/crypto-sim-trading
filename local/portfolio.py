"""
전략별 가상 계좌(포트폴리오) 상태 관리.

각 전략은 독립된 가상 계좌를 가진다:
- 시작 자본: 1000만원 (KRW 기준으로 통일; Binance는 USDT 기준이므로
  환율은 고정 환율(FX_KRW_PER_USDT)로 단순화해서 환산한다)
- 코인 6개 균등 배분: 각 코인에 초기 1000만/6 = 약 166.7만원 할당
- 각 코인 슬롯은 해당 코인에 대해서만 매수/매도 (코인 간 자금 이동 없음,
  "파라미터 자동 재조정" 단계에서 전략 간 자금 재배분을 별도로 다룸)

중요 (아키텍처): 이 모듈은 순수 로직만 담당하고 저장소를 모른다.
상태는 Claude Artifact의 DB(db_write/db_get 등 MCP 도구)에 영속화되며,
그 입출력은 스케줄 세션의 Claude가 직접 수행한다 (engine.py는 DB를 호출하지
않고, "현재 상태 dict -> 갱신된 상태 dict + 실행된 거래 리스트"를 반환하는
순수 함수만 제공한다). to_dict()/from_dict()가 그 경계를 담당한다.
trades는 Portfolio 문서 안에 전체를 담지 않고(문서 크기 제한),
engine.run_cycle()이 반환하는 new_trades만 별도 "trades" 컬렉션에 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Literal

INITIAL_CAPITAL_KRW = 10_000_000.0
COINS = ["BTC", "ETH", "XRP", "SOL", "DOGE", "ADA"]

# Binance는 USDT 기준가라서 KRW 모의투자와 맞추려면 환율이 필요.
# 실거래가 아니므로 고정값으로 단순화 (추후 실시간 환율 API로 교체 가능).
FX_KRW_PER_USDT = 1450.0


@dataclass
class Trade:
    timestamp: str
    coin: str
    side: Literal["BUY", "SELL"]
    price_krw: float
    qty: float
    amount_krw: float
    cash_after: float
    reason: str = ""


@dataclass
class CoinSlot:
    coin: str
    cash_krw: float          # 이 코인에 배정된 현금(매수 대기 자금)
    qty: float = 0.0         # 보유 코인 수량
    avg_cost_krw: float = 0.0  # 평균 매수가(KRW)


@dataclass
class Portfolio:
    strategy_name: str
    initial_capital_krw: float = INITIAL_CAPITAL_KRW
    slots: dict[str, CoinSlot] = field(default_factory=dict)
    # 이번 사이클에서 새로 발생한 거래만 담는다 (영속화되는 전체 로그가 아님).
    # engine.run_cycle()이 채우고, 호출한 쪽이 DB의 별도 "trades" 컬렉션에 쓴다.
    new_trades: list[Trade] = field(default_factory=list)
    trade_count: int = 0  # 지금까지의 총 거래 건수(로그 전체 대신 카운터만 유지)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def new(cls, strategy_name: str, coins: list[str] | None = None,
            initial_capital_krw: float = INITIAL_CAPITAL_KRW) -> "Portfolio":
        coins = coins or COINS
        per_coin = initial_capital_krw / len(coins)
        slots = {c: CoinSlot(coin=c, cash_krw=per_coin) for c in coins}
        return cls(strategy_name=strategy_name, initial_capital_krw=initial_capital_krw, slots=slots)

    # ---- 거래 실행 ----

    def buy(self, coin: str, price_krw: float, pct_of_cash: float = 1.0, reason: str = "") -> bool:
        """해당 코인 슬롯의 현금 중 pct_of_cash 비율만큼 매수. 현금이 없으면 False."""
        slot = self.slots[coin]
        spend = slot.cash_krw * pct_of_cash
        if spend <= 0 or slot.cash_krw <= 0:
            return False
        qty = spend / price_krw
        new_qty = slot.qty + qty
        # 평균단가 갱신
        slot.avg_cost_krw = (
            (slot.avg_cost_krw * slot.qty + spend) / new_qty if new_qty > 0 else 0.0
        )
        slot.qty = new_qty
        slot.cash_krw -= spend
        self.new_trades.append(Trade(
            timestamp=datetime.now(timezone.utc).isoformat(),
            coin=coin, side="BUY", price_krw=price_krw, qty=qty,
            amount_krw=spend, cash_after=slot.cash_krw, reason=reason,
        ))
        self.trade_count += 1
        self.updated_at = datetime.now(timezone.utc).isoformat()
        return True

    def sell(self, coin: str, price_krw: float, pct_of_qty: float = 1.0, reason: str = "") -> bool:
        """해당 코인 보유량 중 pct_of_qty 비율만큼 매도. 보유량 없으면 False."""
        slot = self.slots[coin]
        qty_to_sell = slot.qty * pct_of_qty
        if qty_to_sell <= 0 or slot.qty <= 0:
            return False
        proceeds = qty_to_sell * price_krw
        slot.qty -= qty_to_sell
        slot.cash_krw += proceeds
        if slot.qty <= 1e-12:
            slot.qty = 0.0
            slot.avg_cost_krw = 0.0
        self.new_trades.append(Trade(
            timestamp=datetime.now(timezone.utc).isoformat(),
            coin=coin, side="SELL", price_krw=price_krw, qty=qty_to_sell,
            amount_krw=proceeds, cash_after=slot.cash_krw, reason=reason,
        ))
        self.trade_count += 1
        self.updated_at = datetime.now(timezone.utc).isoformat()
        return True

    # ---- 평가 ----

    def total_value_krw(self, current_prices_krw: dict[str, float]) -> float:
        total = 0.0
        for coin, slot in self.slots.items():
            total += slot.cash_krw
            price = current_prices_krw.get(coin)
            if price is not None:
                total += slot.qty * price
        return total

    def pnl_pct(self, current_prices_krw: dict[str, float]) -> float:
        tv = self.total_value_krw(current_prices_krw)
        return (tv / self.initial_capital_krw - 1.0) * 100

    # ---- DB 문서 변환 (db_write/db_get에 그대로 주고받는 dict) ----

    def to_doc(self) -> dict:
        """DB의 portfolios/<strategy_name> 문서로 쓸 요약 상태.
        new_trades(이번 사이클 거래)는 포함하지 않는다 — 호출한 쪽이
        따로 trades 컬렉션에 쓴다. 문서 크기를 작게 유지하기 위함."""
        return {
            "strategy_name": self.strategy_name,
            "initial_capital_krw": self.initial_capital_krw,
            "slots": {c: asdict(s) for c, s in self.slots.items()},
            "trade_count": self.trade_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_doc(cls, d: dict) -> "Portfolio":
        """db_get으로 읽은 portfolios/<strategy_name> 문서로부터 복원."""
        slots = {c: CoinSlot(**s) for c, s in d["slots"].items()}
        return cls(
            strategy_name=d["strategy_name"],
            initial_capital_krw=d["initial_capital_krw"],
            slots=slots,
            trade_count=d.get("trade_count", 0),
            created_at=d.get("created_at", datetime.now(timezone.utc).isoformat()),
            updated_at=d.get("updated_at", datetime.now(timezone.utc).isoformat()),
        )

    def new_trades_as_dicts(self) -> list[dict]:
        """이번 사이클에서 발생한 거래를 trades 컬렉션 문서들로 변환."""
        return [asdict(t) for t in self.new_trades]


def usdt_to_krw(usdt_price: float, fx: float = FX_KRW_PER_USDT) -> float:
    return usdt_price * fx


def new_empty(strategy_name: str, coins: list[str] | None = None,
              initial_capital_krw: float = INITIAL_CAPITAL_KRW) -> Portfolio:
    """DB에 해당 전략 문서가 아직 없을 때 쓰는 신규 포트폴리오."""
    return Portfolio.new(strategy_name, coins=coins, initial_capital_krw=initial_capital_krw)
