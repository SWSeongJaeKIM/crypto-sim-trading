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

== 숏(공매도) 포지션 ==
바이낸스 선물(USDT-M 퍼프)을 염두에 둔 단순화 모델:
- 고정 레버리지 LEVERAGE배(기본 3배). 숏 진입 시 "마진"으로 현금 일부를
  떼어두고, 그 마진의 LEVERAGE배만큼 숏 수량을 잡는다
  (예: 마진 10만원, 3배 -> 30만원어치 숏 포지션).
- 평가손익(PnL) = (진입가 - 현재가) / 진입가 * 포지션가치. 가격이 떨어지면
  이익, 오르면 손실 — 롱과 정반대.
- 손절(stop-loss): 가격이 진입가 대비 STOP_LOSS_PCT(기본 8%) "불리한" 방향으로
  움직이면(숏은 가격 상승, 롱은 가격 하락) 자동 청산한다. 레버리지가 걸려있어
  실제 마진 손실률은 STOP_LOSS_PCT * LEVERAGE(기본 24%)로 더 크다.
- 강제청산(liquidation): 마진이 완전히 0 이하로 떨어지는 지점
  (가격이 진입가 대비 1/LEVERAGE, 3배면 약 33.3% 불리한 방향)에 도달하면
  마진 전액 손실로 강제 청산한다. 손절(8%)이 그보다 훨씬 먼저 걸리므로
  정상 상황에서는 발생하지 않아야 하는 안전망이고, 체크 누락 등 예외
  상황에서만 작동하는 최종 방어선이다.
- 같은 코인 슬롯에서 롱과 숏을 동시에 들지 않는다: BUY 신호가 들어오면
  숏이 있으면 먼저 청산하고, SELL 신호가 들어오면 롱이 있으면 먼저
  청산한다 (open_long/open_short가 이 전환을 처리).
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

# 숏(공매도) 관련 상수. 바이낸스 선물의 "고정 레버리지 + 격리마진"을 단순화한 모델.
LEVERAGE = 3.0           # 고정 레버리지 (추후 전략별로 다르게 하려면 CoinSlot에 필드 추가)
STOP_LOSS_PCT = 0.08     # 진입가 대비 이 비율(가격 기준)만큼 불리해지면 자동 청산 (롱/숏 동일)
LIQUIDATION_PCT = 1.0 / LEVERAGE  # 마진이 0이 되는 이론적 가격 변동률 (3배 -> 약 33.3%)

# 거래 수수료 (taker 기준 단순화). Upbit 스팟 KRW마켓 taker 수수료(약 0.05%)와
# 바이낸스 USDT-M 선물 taker 수수료(약 0.04~0.05%)가 서로 비슷한 수준이라,
# 롱/숏 구분 없이 하나의 비율로 단순화했다. 지금까지는 이 비용이 전혀
# 반영되지 않아서, "누적" 포트폴리오처럼 리셋 없이 복리로 계속 쌓이는
# 계좌에서는 수수료 없는 매매가 수백~수천 번 반복되며 실제로는 나올 수 없는
# 수익률(예: 1만%대)이 코드상 그대로 가능했다. 정확한 금액은 아니지만,
# "수수료가 전혀 없다"는 가장 비현실적인 가정은 없앤다.
FEE_RATE = 0.0005  # 0.05%, 거래 1건(매수/매도/숏진입/숏청산)의 거래금액(notional) 기준


@dataclass
class Trade:
    timestamp: str
    coin: str
    side: Literal["BUY", "SELL", "SHORT_OPEN", "SHORT_CLOSE", "LONG_LIQUIDATED", "SHORT_LIQUIDATED"]
    price_krw: float
    qty: float
    amount_krw: float
    cash_after: float
    reason: str = ""


@dataclass
class CoinSlot:
    coin: str
    cash_krw: float          # 이 코인에 배정된 현금(매수 대기 자금 + 숏 마진 포함)
    qty: float = 0.0         # 보유 코인 수량 (롱)
    avg_cost_krw: float = 0.0  # 평균 매수가(KRW, 롱) — 손절 기준에는 더 이상 쓰지 않음(아래 참고)
    # 롱 포지션의 "최초" 진입가(KRW). avg_cost_krw는 추가매수마다 가중평균으로
    # 다시 계산되는데, 하락 중 신호가 계속 BUY를 내면 평단가가 가격과 함께
    # 내려가면서 "진입가 대비 STOP_LOSS_PCT 하락" 손절선이 사실상 뒤로 밀리는
    # 물타기형 리스크가 있었다(analysis_team risk 리포트 2026-11-22_0417 발견점
    # #1, synthesizer Top3 #1). 포지션을 처음 열 때의 가격을 별도로 고정해두고
    # 손절 판단은 이 값을 기준으로 한다 — 평단가는 대시보드 표시 등 기존 용도에
    # 계속 쓰인다. 포지션이 0으로 완전히 청산되면 다음 재진입을 위해 0으로
    # 리셋한다(새 "최초" 진입이 되므로).
    first_entry_price_krw: float = 0.0
    short_qty: float = 0.0        # 숏 포지션 수량 (코인 단위, 항상 0 이상)
    short_avg_entry_krw: float = 0.0  # 숏 평균 진입가(KRW) — 손절 기준에는 더 이상 쓰지 않음
    # 숏 포지션의 "최초" 진입가(KRW). 롱의 first_entry_price_krw와 동일한 이유로,
    # 숏도 추가 진입마다 평균 진입가가 갱신되면 같은 물타기형 리스크가 대칭적으로
    # 생긴다(가격이 계속 올라도 평균 진입가가 같이 올라가 손절선이 뒤로 밀림).
    short_first_entry_price_krw: float = 0.0
    short_margin_krw: float = 0.0     # 숏에 걸어둔 마진(현금에서 분리해 보관 중인 금액)


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
    # pnl_pct()만으로는 "수익률"만 보이고 변동성/낙폭 같은 리스크 지표가
    # 전혀 없어, 변동성이 큰 코인에 쏠린 전략이 우연히 좋은 성과로만 보일 수
    # 있었다(analysis_team risk 리포트 2026-11-22_0918 발견점 #4, synthesizer
    # Top3 #2). peak_value_krw(지금까지 관측된 평가금액 최고치)와
    # max_drawdown_pct(그 고점 대비 최대 하락률)를 추가해 to_doc()에 노출한다.
    # 기존에 저장된 포트폴리오 문서에는 이 필드가 없지만, 아래 둘 다 기본값
    # 0.0으로 from_doc()에서 자연히 채워지므로 마이그레이션이 필요 없다 —
    # 다만 그 경우 "지금까지의" 최고치가 아니라 "이 필드 추가 이후"의 최고치부터
    # 다시 추적을 시작하게 된다(과거 데이터를 추정해서 채우지 않음).
    peak_value_krw: float = 0.0
    max_drawdown_pct: float = 0.0
    # 매 거래마다 계산되는 fee는 지금까지 Trade.reason 문자열에만 남고 집계되지
    # 않아, 장기 성과 저하가 "신호 품질 문제"인지 "수수료 누적 문제"인지 구분할
    # 지표가 없었다(analysis_team risk 리포트 2026-11-23_0117 발견점 #4,
    # synthesizer Top3 #2). buy/sell/open_short/close_short 각각에서 누적한다.
    # peak_value_krw와 동일한 이유로 이 필드 추가 이후의 수수료만 정확하다
    # (과거 거래분은 거슬러 계산하지 않음 — 마이그레이션 불필요).
    total_fees_krw: float = 0.0

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
        fee = spend * FEE_RATE
        qty = spend / price_krw
        if slot.qty <= 0:
            # 새로 여는(또는 완전히 비어 있던) 포지션의 "최초" 진입가를 고정.
            # 이후 추가매수로 avg_cost_krw가 바뀌어도 이 값은 포지션이 완전히
            # 청산되기 전까지 유지된다.
            slot.first_entry_price_krw = price_krw
        new_qty = slot.qty + qty
        # 평균단가 갱신 (수수료는 손절 기준선에는 영향 주지 않도록 cost basis에는 포함하지 않음)
        slot.avg_cost_krw = (
            (slot.avg_cost_krw * slot.qty + spend) / new_qty if new_qty > 0 else 0.0
        )
        slot.qty = new_qty
        slot.cash_krw -= spend
        slot.cash_krw -= fee
        self.total_fees_krw += fee
        self.new_trades.append(Trade(
            timestamp=datetime.now(timezone.utc).isoformat(),
            coin=coin, side="BUY", price_krw=price_krw, qty=qty,
            amount_krw=spend, cash_after=slot.cash_krw,
            reason=f"{reason} [수수료 {fee:,.0f}원]",
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
        fee = proceeds * FEE_RATE
        slot.qty -= qty_to_sell
        slot.cash_krw += proceeds - fee
        self.total_fees_krw += fee
        if slot.qty <= 1e-12:
            slot.qty = 0.0
            slot.avg_cost_krw = 0.0
            slot.first_entry_price_krw = 0.0
        self.new_trades.append(Trade(
            timestamp=datetime.now(timezone.utc).isoformat(),
            coin=coin, side="SELL", price_krw=price_krw, qty=qty_to_sell,
            amount_krw=proceeds, cash_after=slot.cash_krw,
            reason=f"{reason} [수수료 {fee:,.0f}원]",
        ))
        self.trade_count += 1
        self.updated_at = datetime.now(timezone.utc).isoformat()
        return True

    # ---- 숏(공매도) ----

    def open_short(self, coin: str, price_krw: float, pct_of_cash: float = 1.0, reason: str = "") -> bool:
        """해당 코인 슬롯의 현금 중 pct_of_cash 비율을 마진으로 잡아 숏 진입(또는 추가).
        LEVERAGE배만큼의 가상 포지션을 잡는다. 현금이 없으면 False."""
        slot = self.slots[coin]
        margin = slot.cash_krw * pct_of_cash
        if margin <= 0 or slot.cash_krw <= 0:
            return False
        notional = margin * LEVERAGE
        fee = notional * FEE_RATE  # 선물 수수료는 마진이 아니라 포지션 전체(notional) 기준
        qty = notional / price_krw
        if slot.short_qty <= 0:
            # 롱의 first_entry_price_krw와 동일한 이유로 숏의 "최초" 진입가도
            # 고정한다 (아래 short_first_entry_price_krw 참고).
            slot.short_first_entry_price_krw = price_krw
        new_qty = slot.short_qty + qty
        # 평균 진입가 갱신 (추가 진입 시 가중평균)
        slot.short_avg_entry_krw = (
            (slot.short_avg_entry_krw * slot.short_qty + price_krw * qty) / new_qty
            if new_qty > 0 else 0.0
        )
        slot.short_qty = new_qty
        slot.short_margin_krw += margin
        slot.cash_krw -= margin
        slot.cash_krw -= fee
        self.total_fees_krw += fee
        self.new_trades.append(Trade(
            timestamp=datetime.now(timezone.utc).isoformat(),
            coin=coin, side="SHORT_OPEN", price_krw=price_krw, qty=qty,
            amount_krw=margin, cash_after=slot.cash_krw,
            reason=f"{reason} [레버리지 {LEVERAGE:.0f}배, 마진 {margin:,.0f}원, 수수료 {fee:,.0f}원]",
        ))
        self.trade_count += 1
        self.updated_at = datetime.now(timezone.utc).isoformat()
        return True

    def close_short(self, coin: str, price_krw: float, pct_of_qty: float = 1.0,
                     reason: str = "", side_label: str = "SHORT_CLOSE") -> bool:
        """숏 포지션 중 pct_of_qty 비율만큼 청산. 마진 + 평가손익을 현금으로 되돌린다.
        손익 = (진입가 - 청산가) * 수량 (가격이 떨어지면 이익, 오르면 손실).
        마진 전액 손실(강제청산)이면 cash_krw가 그만큼 그대로 줄어든다."""
        slot = self.slots[coin]
        qty_to_close = slot.short_qty * pct_of_qty
        if qty_to_close <= 0 or slot.short_qty <= 0:
            return False
        margin_released = slot.short_margin_krw * pct_of_qty
        pnl = (slot.short_avg_entry_krw - price_krw) * qty_to_close
        fee = (qty_to_close * price_krw) * FEE_RATE  # 청산 시점 notional 기준 수수료
        returned = margin_released + pnl - fee
        # 레버리지 포지션이라 이론상 손실(+수수료)이 마진을 초과할 수 있다 —
        # 격리마진 방식이므로 반환액이 0 밑으로 내려가지 않게(추가 손실을
        # 현금에서 더 떼어가지 않게) 막는다. 정상적으로는 손절이 마진 전액
        # 손실보다 훨씬 먼저 걸리므로(STOP_LOSS_PCT < LIQUIDATION_PCT) 이
        # 경로는 check_stop_loss_and_liquidation()이 손절을 놓쳤을 때만 닿는다.
        returned = max(0.0, returned)
        slot.short_qty -= qty_to_close
        slot.short_margin_krw -= margin_released
        slot.cash_krw += returned
        self.total_fees_krw += fee
        if slot.short_qty <= 1e-12:
            slot.short_qty = 0.0
            slot.short_avg_entry_krw = 0.0
            slot.short_margin_krw = 0.0
            slot.short_first_entry_price_krw = 0.0
        self.new_trades.append(Trade(
            timestamp=datetime.now(timezone.utc).isoformat(),
            coin=coin, side=side_label, price_krw=price_krw, qty=qty_to_close,
            amount_krw=returned, cash_after=slot.cash_krw,
            reason=f"{reason} [손익 {pnl:+,.0f}원, 수수료 {fee:,.0f}원]",
        ))
        self.trade_count += 1
        self.updated_at = datetime.now(timezone.utc).isoformat()
        return True

    # ---- 신호 해석 (롱/숏 전환) ----
    # strategies.py/lstm_model.py는 BUY/SELL/HOLD만 낸다 — 숏 개념을 몰라도
    # 된다. 여기서 "BUY는 숏을 정리하고 롱 방향, SELL은 롱을 정리하고 숏
    # 방향"으로 해석해 같은 코인 슬롯에 롱/숏을 동시에 들지 않게 한다.

    def apply_buy_signal(self, coin: str, price_krw: float, fraction: float, reason: str = "") -> None:
        slot = self.slots[coin]
        if slot.short_qty > 0:
            self.close_short(coin, price_krw, pct_of_qty=1.0, reason=f"{reason} (숏 정리 후 롱 전환)")
        self.buy(coin, price_krw, pct_of_cash=fraction, reason=reason)

    def apply_sell_signal(self, coin: str, price_krw: float, fraction: float, reason: str = "") -> None:
        slot = self.slots[coin]
        if slot.qty > 0:
            self.sell(coin, price_krw, pct_of_qty=1.0, reason=f"{reason} (롱 정리 후 숏 전환)")
        # open_short()의 pct_of_cash는 "마진" 비율이고, open_short 내부에서
        # notional = margin * LEVERAGE로 LEVERAGE배를 곱한다. fraction은
        # confidence_to_trade_fraction()이 만든 0.1~0.9 비율로 buy()의
        # pct_of_cash(현금 중 비율 = 그대로 notional 비율)와 동일 스케일이라,
        # 레버리지를 감안하지 않고 그대로 넘기면 같은 확신도에서 숏 notional이
        # 롱보다 LEVERAGE배 커진다 (손절 시 손실도 LEVERAGE배). fraction을
        # LEVERAGE로 나눠 margin*LEVERAGE(=notional)가 다시 cash*fraction이
        # 되도록 맞춰, 같은 확신도에서 롱/숏의 자금 노출(notional)을 동일하게
        # 만든다 (analysis_team 리포트 2026-11-22_1418_risk.md 발견점 #1,
        # synthesizer Top3 #2).
        self.open_short(coin, price_krw, pct_of_cash=fraction / LEVERAGE, reason=reason)

    # ---- 손절 / 강제청산 ----
    # main_loop.py가 매 사이클(1분)마다 모든 코인에 대해 호출해야 한다.
    # 신호(BUY/SELL)와 무관하게, 지금 들고 있는 포지션이 손절선을 넘었는지
    # 가격만 보고 매 사이클 확인한다.

    def check_stop_loss_and_liquidation(self, coin: str, price_krw: float) -> str | None:
        """현재가 기준으로 롱/숏 포지션의 손절 또는 강제청산 조건을 확인하고,
        필요하면 즉시 청산한다. 청산이 일어났으면 그 사유 문자열을, 아니면
        None을 반환한다 (대시보드/로그용).

        주의: 아래에서 남기는 reason 문자열이 "손절" 또는 "강제청산"으로
        시작하는 접두어는 storage.py의 stop_loss_counts_by_coin()이 문자열
        매칭(LIKE '손절%'/'강제청산%')으로 그대로 의존한다. 이 접두어를
        바꾸면 그 집계 함수도 같이 수정해야 한다 — 안 그러면 에러 없이
        조용히 0건으로 집계되어 "손절이 전혀 없다"는 잘못된 결론을 낼 수
        있다 (storage.py의 stop_loss_counts_by_coin() 쪽에 대칭되는 설명이
        있음, analysis_team storage 리포트 2026-11-22_1219 발견점 #3)."""
        slot = self.slots[coin]

        if slot.qty > 0:
            # first_entry_price_krw(최초 진입가)를 기준으로 판단한다. 기존
            # DB에 저장된 포지션(이 필드 추가 전에 이미 열려 있던 포지션)은
            # 0.0으로 로드되므로, 그 경우에만 이전 동작(avg_cost_krw 기준)으로
            # 폴백한다 — 다음 번 완전 청산 후 재진입부터는 first_entry_price_krw가
            # 채워져 제대로 적용된다.
            entry_price = slot.first_entry_price_krw if slot.first_entry_price_krw > 0 else slot.avg_cost_krw
            if entry_price > 0:
                loss_pct = (entry_price - price_krw) / entry_price
                if loss_pct >= STOP_LOSS_PCT:
                    self.sell(coin, price_krw, pct_of_qty=1.0,
                              reason=f"손절 (진입가 대비 {loss_pct*100:.1f}% 하락)")
                    return f"롱 손절: {coin} 진입가 대비 {loss_pct*100:.1f}% 하락"

        if slot.short_qty > 0:
            entry_price = (
                slot.short_first_entry_price_krw if slot.short_first_entry_price_krw > 0
                else slot.short_avg_entry_krw
            )
            if entry_price <= 0:
                return None
            loss_pct = (price_krw - entry_price) / entry_price
            if loss_pct >= LIQUIDATION_PCT:
                # 안전망: 마진 전액 손실 지점. 손절(STOP_LOSS_PCT)이 항상 더
                # 먼저 걸려야 하므로 정상 상황에서는 이 분기가 아니라 바로
                # 아래 손절 분기에서 먼저 청산된다.
                self.close_short(coin, price_krw, pct_of_qty=1.0,
                                  reason=f"강제청산 (진입가 대비 {loss_pct*100:.1f}% 상승, 마진 소진)",
                                  side_label="SHORT_LIQUIDATED")
                return f"숏 강제청산: {coin} 진입가 대비 {loss_pct*100:.1f}% 상승"
            if loss_pct >= STOP_LOSS_PCT:
                self.close_short(coin, price_krw, pct_of_qty=1.0,
                                  reason=f"손절 (진입가 대비 {loss_pct*100:.1f}% 상승)")
                return f"숏 손절: {coin} 진입가 대비 {loss_pct*100:.1f}% 상승"

        return None

    # ---- 평가 ----

    def slot_values_krw(self, current_prices_krw: dict[str, float]) -> dict[str, float]:
        """코인별 슬롯 평가금액(현금 + 롱 평가금액 + 숏 마진/미실현손익).

        total_value_krw()가 내부에서 코인별로 계산하던 값을 그대로 분리한
        것으로, 계산식은 완전히 동일하다 — cumulative 포트폴리오(코인 간
        자금 이동이 없는 구조)에서 "장기 성과가 실제로 6개 코인에 분산된
        결과인지, 특정 1~2개 코인에만 좌우된 결과인지"를 보여줄 지표가
        없다는 점을 보완하기 위한 조회용 함수다(analysis_team risk 리포트
        2026-11-22_1818 발견점 #4, 구체적 제안 #3). DB에는 저장하지 않고
        호출 시점 가격으로 즉석 계산만 하므로(저장된 값이 과거 가격에
        묶이는 문제 없음), 배분/손절선 등 매매 판단에는 전혀 영향을
        주지 않는다."""
        values = {}
        for coin, slot in self.slots.items():
            total = slot.cash_krw
            price = current_prices_krw.get(coin)
            if price is not None:
                total += slot.qty * price
                if slot.short_qty > 0:
                    unrealized_pnl = (slot.short_avg_entry_krw - price) * slot.short_qty
                    total += slot.short_margin_krw + unrealized_pnl
            else:
                # 가격을 모르면 숏 평가는 못 하지만, 최소한 마진(원가)은 더해서
                # 그 금액이 통째로 사라진 것처럼 보이지 않게 한다.
                total += slot.short_margin_krw
            values[coin] = total
        return values

    def total_value_krw(self, current_prices_krw: dict[str, float]) -> float:
        """현금 + 롱 평가금액 + (숏 마진 + 숏 미실현손익). cash_krw에는 숏을
        열 때 떼어둔 마진이 빠져 있으므로, 마진과 그 위의 미실현손익을 다시
        더해줘야 전체 평가금액이 맞다. 미실현손익이 마진을 넘는 음수가
        되더라도(강제청산 전) 여기서는 그대로 보여준다 — 실제 마진 초과
        손실 제한은 close_short()에서, 손절 체크는 check_stop_loss_and_
        liquidation()에서 별도로 처리.

        코인별 세부값은 slot_values_krw()로 분리되어 있고, 이 함수는 그
        합을 반환할 뿐 계산식이 바뀌지는 않았다."""
        return sum(self.slot_values_krw(current_prices_krw).values())

    def pnl_pct(self, current_prices_krw: dict[str, float]) -> float:
        tv = self.total_value_krw(current_prices_krw)
        return (tv / self.initial_capital_krw - 1.0) * 100

    def update_drawdown(self, total_value_krw: float) -> float:
        """현재 평가금액(호출한 쪽이 total_value_krw()로 이미 계산해둔 값을
        그대로 넘긴다 — 여기서 다시 계산하지 않음)을 기준으로 peak_value_krw를
        갱신하고, 고점 대비 낙폭(max_drawdown_pct, 0 이상 %)을 갱신한다.
        매 사이클 호출해야 다음 사이클의 peak 판단이 정확하다 — 호출한 쪽
        (main_loop.py)이 to_doc() 저장 직전에 호출한다. 갱신된
        max_drawdown_pct를 반환한다(호출부에서 바로 로그에 쓸 수 있도록)."""
        if total_value_krw > self.peak_value_krw:
            self.peak_value_krw = total_value_krw
        if self.peak_value_krw > 0:
            drawdown_pct = max(0.0, (self.peak_value_krw - total_value_krw) / self.peak_value_krw * 100)
            if drawdown_pct > self.max_drawdown_pct:
                self.max_drawdown_pct = drawdown_pct
        return self.max_drawdown_pct

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
            "peak_value_krw": self.peak_value_krw,
            "max_drawdown_pct": self.max_drawdown_pct,
            "total_fees_krw": self.total_fees_krw,
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
            peak_value_krw=d.get("peak_value_krw", 0.0),
            max_drawdown_pct=d.get("max_drawdown_pct", 0.0),
            total_fees_krw=d.get("total_fees_krw", 0.0),
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
