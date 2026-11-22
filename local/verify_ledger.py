"""
trading.db 거래 원장(ledger) 무결성 검증 스크립트.

이 스크립트는 사용자 PC에서 돌고 있는 local/storage.py, local/portfolio.py와
"같은 디렉터리"에 두고 실행한다 (수수료/레버리지/손절 임계값 등 상수를 그
모듈에서 그대로 가져와 쓰기 때문에, 그 값들이 나중에 바뀌어도 이 스크립트를
따로 고칠 필요가 최소화된다).

사용법 (local/ 디렉터리에서):
    python3 verify_ledger.py                      # 전체 전략 x 모드(reset/cumulative) 검사
    python3 verify_ledger.py --strategy ma_cross_rsi --mode cumulative   # 특정 하나만
    python3 verify_ledger.py --db /path/to/trading.db   # DB 경로를 직접 지정

무엇을, 왜 검증하는가
----------------------
"특정 매매 방법의 수익이 말이 안 된다"는 의심은 크게 두 갈래로 나뉜다:
  (a) 거래 자체가 정상적으로 기록/반영됐는가? (버그, 쓰기 누락, 이중 기록 등)
  (b) 그 거래들이 실제로 "말이 되는" 시장 상황에서 나온 것인가?
이 스크립트는 (a)만 다룬다 — (b)(그 시각 실제 시세와 맞는지, look-ahead
bias가 있는지)는 Upbit의 그 시각 분봉 데이터가 추가로 필요해서 이 스크립트
범위 밖이다(결과 출력 끝에 다시 안내).

trades 테이블에는 side/qty/price_krw/reason/timestamp만 저장되고(참고:
portfolio.py의 Trade 데이터클래스에 있는 amount_krw/cash_after 필드는
storage.add_trades()가 DB에 쓸 때 빠진다 — trades 테이블 스키마 자체에
그 두 컬럼이 없다), "그 시점 cash_after가 맞았는지"를 직접 대조할 수는
없다. 대신 qty와 price_krw만으로 portfolio.py의 buy()/sell()/open_short()/
close_short() 공식을 처음(초기자본)부터 그대로 재연산(시뮬레이션)해서
cash/보유량을 코인별로 복원하고, 그 결과를 지금 저장된 최종 상태와 대조한다.

1. 코인별 전체 재시뮬레이션 — trades를 시간순으로 재생하며 코인별
   cash_krw/보유수량(long qty)/숏수량/숏평단가/숏마진을 처음부터 다시
   계산한다. "reset" 모드는 RESET_DAYS마다 포트폴리오가 초기자본으로
   리셋되는데 trades 테이블에는 리셋 "이벤트"가 따로 기록되지 않으므로,
   reset 모드는 현재 포트폴리오 문서의 created_at(이번 리셋 주기가 시작된
   시각) 이후의 거래만 재생한다 — 그 전 주기는 이미 끝난 과거라 검증
   대상에서 뺀다. cumulative 모드는 리셋이 없으므로 전체 역사를 재생한다.
2. 최종 상태 대조 — 재시뮬레이션 결과(코인별 cash/수량/숏수량)가 지금
   portfolios 테이블에 저장된 최신 문서와 일치하는지 비교한다.
   -> 일치하지 않으면 "거래 기록과 포트폴리오 문서가 서로 안 맞는다"는
      뜻으로, 쓰기 누락/동시성 버그/수동 조작 등을 의심할 근거가 된다.
   -> 일치하면 최소한 "지금 저장된 수익률(pnl_pct)은 기록된 거래들만으로
      설명 가능한 값"이라는 뜻이다(즉 거래 기록 외의 경로로 돈이 생기거나
      사라지지 않았다는 뜻). 그 수익률 자체가 시장 상황에 비해 말이
      안 되는지는 (b) 영역이라 이 스크립트가 "정상"이라고 보증하는 건
      아니다.
3. 보유량 초과 매도/청산 탐지 — 재생 중 어떤 SELL/SHORT_CLOSE가 그 시점
   시뮬레이션상 가진 수량보다 많이 팔려고 하면(정상 로직에서는 일어날 수
   없는 일) 이상치로 표시한다.
4. 손절/강제청산 reason 임계값 재확인 — reason이 "손절"/"강제청산"으로
   시작하는 행에 적힌 퍼센트가 실제로 STOP_LOSS_PCT(8%)/LIQUIDATION_PCT
   (약 33.3%) 이상인지 재확인한다.
5. 중복 거래 탐지 — (strategy, coin, side, qty, price_krw, timestamp)가
   완전히 똑같은 행이 2건 이상이면 이중 기록 가능성으로 표시한다.
6. 비정상 수익률 구간 플래그 — snapshots의 pnl_pct가 바로 다음 스냅샷으로
   넘어가면서 한 번에 --jump-threshold(기본 50)%p 이상 뛴 지점을 찾아
   "사람이 들여다볼 구간"으로 표시한다(버그 확정이 아니라 우선 조사 순서
   제안용 — 고배율 숏/고확신 베팅은 실제로 한 사이클에 크게 움직일 수 있다).

주의: 이 스크립트는 SELECT만 수행하며(읽기 전용 모드로 연결), trading.db를
전혀 수정하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import portfolio as portfolio_mod
    FEE_RATE = portfolio_mod.FEE_RATE
    LEVERAGE = portfolio_mod.LEVERAGE
    STOP_LOSS_PCT = portfolio_mod.STOP_LOSS_PCT
    LIQUIDATION_PCT = portfolio_mod.LIQUIDATION_PCT
    INITIAL_CAPITAL_KRW = portfolio_mod.INITIAL_CAPITAL_KRW
    COINS = portfolio_mod.COINS
except Exception as e:  # pragma: no cover - 표준 배치 위치가 아닐 때의 안전망
    print(f"[경고] local/portfolio.py를 import하지 못해 하드코딩된 기본값을 씁니다: {e}",
          file=sys.stderr)
    FEE_RATE = 0.0005
    LEVERAGE = 3.0
    STOP_LOSS_PCT = 0.08
    LIQUIDATION_PCT = 1.0 / 3.0
    INITIAL_CAPITAL_KRW = 10_000_000.0
    COINS = ["BTC", "ETH", "XRP", "SOL", "DOGE", "ADA"]

try:
    import storage as storage_mod
    DEFAULT_DB_PATH = storage_mod.DB_PATH
    parse_portfolio_key = storage_mod.parse_portfolio_key
except Exception as e:  # pragma: no cover
    print(f"[경고] local/storage.py를 import하지 못해 기본 경로/파서를 자체 구현합니다: {e}",
          file=sys.stderr)
    DEFAULT_DB_PATH = Path(__file__).resolve().parent / "trading.db"

    def parse_portfolio_key(key: str) -> tuple[str, str]:
        strategy, sep, variant = key.rpartition("__")
        if not sep or variant not in ("reset", "cumulative"):
            return key, "cumulative"
        return strategy, variant


PER_COIN_INITIAL = INITIAL_CAPITAL_KRW / len(COINS)
QTY_EPS = 1e-8
CASH_TOL = 1.0  # 원 단위 허용 오차 (부동소수점 누적 오차용)


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


class CoinState:
    __slots__ = ("cash", "long_qty", "short_qty", "short_avg_entry", "short_margin")

    def __init__(self, cash: float):
        self.cash = cash
        self.long_qty = 0.0
        self.short_qty = 0.0
        self.short_avg_entry = 0.0
        self.short_margin = 0.0


def simulate_coin(rows: list[sqlite3.Row]) -> tuple[CoinState, list[dict]]:
    """portfolio.py의 buy()/sell()/open_short()/close_short() 공식을 그대로
    재현해, trades 행(qty, price_krw, side)만으로 코인 하나의 상태를
    처음(PER_COIN_INITIAL)부터 재구성한다. (state, anomalies) 반환."""
    st = CoinState(PER_COIN_INITIAL)
    anomalies = []
    for r in rows:
        side = r["side"]
        qty = r["qty"]
        price = r["price_krw"]

        if side == "BUY":
            spend = qty * price
            fee = spend * FEE_RATE
            st.cash -= spend + fee
            st.long_qty += qty

        elif side == "SELL":
            if qty > st.long_qty + max(QTY_EPS, st.long_qty * 1e-6):
                anomalies.append({
                    "id": r["id"], "coin": r["coin"], "timestamp": r["timestamp"],
                    "note": f"SELL 수량({qty})이 재생 중 보유량({st.long_qty})을 초과",
                })
            proceeds = qty * price
            fee = proceeds * FEE_RATE
            st.cash += proceeds - fee
            st.long_qty = max(0.0, st.long_qty - qty)
            if st.long_qty <= QTY_EPS:
                st.long_qty = 0.0

        elif side == "SHORT_OPEN":
            notional = qty * price
            margin = notional / LEVERAGE
            fee = notional * FEE_RATE
            new_short_qty = st.short_qty + qty
            if new_short_qty > 0:
                st.short_avg_entry = (
                    (st.short_avg_entry * st.short_qty + price * qty) / new_short_qty
                )
            st.short_qty = new_short_qty
            st.short_margin += margin
            st.cash -= margin + fee

        elif side in ("SHORT_CLOSE", "SHORT_LIQUIDATED"):
            if qty > st.short_qty + max(QTY_EPS, st.short_qty * 1e-6):
                anomalies.append({
                    "id": r["id"], "coin": r["coin"], "timestamp": r["timestamp"],
                    "note": f"{side} 수량({qty})이 재생 중 숏보유량({st.short_qty})을 초과",
                })
            pct = (qty / st.short_qty) if st.short_qty > 0 else 0.0
            pct = min(pct, 1.0)
            margin_released = st.short_margin * pct
            pnl = (st.short_avg_entry - price) * qty
            fee = (qty * price) * FEE_RATE
            returned = max(0.0, margin_released + pnl - fee)
            st.cash += returned
            st.short_qty = max(0.0, st.short_qty - qty)
            st.short_margin -= margin_released
            if st.short_qty <= QTY_EPS:
                st.short_qty = 0.0
                st.short_avg_entry = 0.0
                st.short_margin = 0.0

        else:
            anomalies.append({
                "id": r["id"], "coin": r["coin"], "timestamp": r["timestamp"],
                "note": f"알 수 없는 side: {side!r}",
            })

    return st, anomalies


def find_duplicates(rows: list[sqlite3.Row]) -> list[dict]:
    seen = defaultdict(int)
    for r in rows:
        sig = (r["coin"], r["side"], round(r["qty"], 10), round(r["price_krw"], 6), r["timestamp"])
        seen[sig] += 1
    return [{"signature": sig, "count": cnt} for sig, cnt in seen.items() if cnt > 1]


_REASON_PCT_RE = re.compile(r"(손절|강제청산)\s*\(진입가 대비\s*([\d.]+)%")


def check_stop_loss_reasons(rows: list[sqlite3.Row]) -> list[dict]:
    issues = []
    for r in rows:
        reason = r["reason"] or ""
        if not (reason.startswith("손절") or reason.startswith("강제청산")):
            continue
        m = _REASON_PCT_RE.search(reason)
        if not m:
            issues.append({"id": r["id"], "coin": r["coin"], "timestamp": r["timestamp"],
                            "note": f"reason에서 퍼센트를 못 찾음: {reason!r}"})
            continue
        kind, pct_str = m.group(1), m.group(2)
        pct = float(pct_str) / 100.0
        threshold = LIQUIDATION_PCT if kind == "강제청산" else STOP_LOSS_PCT
        if pct < threshold - 1e-9:
            issues.append({
                "id": r["id"], "coin": r["coin"], "timestamp": r["timestamp"],
                "note": f"{kind} reason인데 기록된 비율({pct*100:.2f}%)이 임계값"
                        f"({threshold*100:.2f}%)에 못 미침",
            })
    return issues


def check_snapshot_jumps(conn: sqlite3.Connection, key: str, jump_threshold: float) -> list[dict]:
    rows = conn.execute(
        "SELECT timestamp, pnl_pct FROM snapshots WHERE strategy = ? ORDER BY timestamp ASC",
        (key,),
    ).fetchall()
    flags = []
    prev = None
    for r in rows:
        if prev is not None:
            delta = r["pnl_pct"] - prev["pnl_pct"]
            if abs(delta) >= jump_threshold:
                flags.append({
                    "from_timestamp": prev["timestamp"], "to_timestamp": r["timestamp"],
                    "from_pnl_pct": round(prev["pnl_pct"], 2), "to_pnl_pct": round(r["pnl_pct"], 2),
                    "delta_pp": round(delta, 2),
                })
        prev = r
    return flags


def list_all_keys(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT DISTINCT strategy FROM portfolios ORDER BY strategy").fetchall()
    return [r["strategy"] for r in rows]


def analyze_key(conn: sqlite3.Connection, key: str, jump_threshold: float = 50.0) -> dict:
    """하나의 portfolios 키에 대해 모든 검증을 수행하고 구조화된 dict로 반환한다.
    CLI(run_for_key)와 dashboard.py의 /api/verify 라우트가 이 함수 하나를
    공유해서 쓴다 — 검증 로직이 두 군데서 따로 구현되어 서로 달라지는 걸 막는다."""
    strategy_name, variant = parse_portfolio_key(key)

    doc_row = conn.execute("SELECT data_json FROM portfolios WHERE strategy = ?", (key,)).fetchone()
    if doc_row is None:
        return {
            "key": key, "strategy": strategy_name, "variant": variant,
            "found": False, "note": "portfolios 테이블에 이 키의 문서가 없음",
        }
    doc = json.loads(doc_row["data_json"])

    all_rows = conn.execute(
        "SELECT * FROM trades WHERE strategy = ? ORDER BY timestamp ASC, id ASC",
        (key,),
    ).fetchall()

    if variant == "reset":
        created_at = doc.get("created_at", "")
        sim_rows = [r for r in all_rows if r["timestamp"] >= created_at]
        scope_note = (f"이번 주기 시작(created_at={created_at}) 이후 거래만 검증 대상 "
                       f"({len(sim_rows)}건) — 그 전 주기들은 이미 끝난 과거라 범위 밖")
    else:
        created_at = doc.get("created_at")
        sim_rows = all_rows
        scope_note = f"전체 역사({len(sim_rows)}건)를 검증 대상으로 함"

    dups = find_duplicates(all_rows)

    by_coin: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for r in sim_rows:
        by_coin[r["coin"]].append(r)

    slots = doc.get("slots", {})
    all_anomalies: list[dict] = []
    recon_issues: list[dict] = []
    for coin in COINS:
        coin_rows = by_coin.get(coin, [])
        st, anomalies = simulate_coin(coin_rows)
        all_anomalies.extend(anomalies)

        slot = slots.get(coin)
        if slot is None:
            recon_issues.append({"coin": coin, "note": "portfolios 문서에 이 코인 슬롯이 없음"})
            continue
        for label, sim_val, doc_val, tol in [
            ("cash_krw", st.cash, slot.get("cash_krw", 0.0), CASH_TOL),
            ("qty", st.long_qty, slot.get("qty", 0.0), 1e-6),
            ("short_qty", st.short_qty, slot.get("short_qty", 0.0), 1e-6),
        ]:
            if abs(sim_val - doc_val) > tol:
                recon_issues.append({
                    "coin": coin, "field": label,
                    "replayed": round(sim_val, 6), "stored": round(doc_val, 6),
                    "diff": round(sim_val - doc_val, 6),
                })

    sl_issues = check_stop_loss_reasons(all_rows)
    jump_flags = check_snapshot_jumps(conn, key, jump_threshold)

    ok = not (dups or all_anomalies or recon_issues or sl_issues)
    return {
        "key": key, "strategy": strategy_name, "variant": variant, "found": True,
        "created_at": created_at, "n_trades_total": len(all_rows), "n_trades_checked": len(sim_rows),
        "scope_note": scope_note, "ok": ok,
        "duplicates": dups, "qty_anomalies": all_anomalies, "reconciliation_issues": recon_issues,
        "stop_loss_issues": sl_issues, "pnl_jump_flags": jump_flags,
    }


def run_for_key(conn: sqlite3.Connection, key: str, jump_threshold: float) -> None:
    r = analyze_key(conn, key, jump_threshold)
    print(f"\n{'=' * 70}")
    print(f"[{r['key']}]  (strategy={r['strategy']}, variant={r['variant']})")
    print(f"{'=' * 70}")

    if not r["found"]:
        print(f"- {r['note']} — 건너뜀")
        return

    print(f"- 전체 거래 수(이 키 역사 전체): {r['n_trades_total']}")
    print(f"- {r['scope_note']}")

    dups = r["duplicates"]
    if dups:
        print(f"- [중복 거래 의심] {len(dups)}건:")
        for d in dups[:10]:
            print(f"    {d['signature']} x{d['count']}")
    else:
        print("- 중복 거래: 없음")

    all_anomalies = r["qty_anomalies"]
    if all_anomalies:
        print(f"- [보유량 초과 매도/청산] {len(all_anomalies)}건 — 정상 로직에서는 일어날 수 없는 패턴:")
        for a in all_anomalies[:20]:
            print(f"    {a}")
    else:
        print("- 보유량 초과 매도/청산: 없음")

    recon_issues = r["reconciliation_issues"]
    if recon_issues:
        print(f"- [최종 상태 대조 불일치] {len(recon_issues)}건 — 거래기록과 포트폴리오 문서가 서로 안 맞음:")
        for i in recon_issues:
            print(f"    {i}")
    else:
        scope = "이번 리셋 주기" if r["variant"] == "reset" else "전체 역사"
        print(f"- 최종 상태(cash/qty/short_qty) 대조: {scope} 거래 기록만으로 지금 저장된 상태가 "
              f"그대로 설명됨 (버그 신호 없음)")

    sl_issues = r["stop_loss_issues"]
    if sl_issues:
        print(f"- [손절/강제청산 reason 임계값 재확인] {len(sl_issues)}건 의심:")
        for i in sl_issues[:10]:
            print(f"    {i}")
    else:
        print("- 손절/강제청산 reason 임계값: 전부 정상")

    jump_flags = r["pnl_jump_flags"]
    if jump_flags:
        print(f"- [수익률 급변 구간] {len(jump_flags)}건 ({jump_threshold}%p 이상, 수동 확인 권장):")
        for j in jump_flags[:10]:
            print(f"    {j['from_timestamp']} ({j['from_pnl_pct']}%) -> "
                  f"{j['to_timestamp']} ({j['to_pnl_pct']}%)  Δ={j['delta_pp']}%p")
    else:
        print(f"- 수익률 급변 구간({jump_threshold}%p 이상): 없음")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=str, default=str(DEFAULT_DB_PATH), help="trading.db 경로")
    ap.add_argument("--strategy", type=str, default=None, help="특정 전략 이름만 (예: ma_cross_rsi)")
    ap.add_argument("--mode", type=str, default=None, choices=["reset", "cumulative"],
                     help="특정 모드만 (strategy와 함께 쓸 때만 의미 있음)")
    ap.add_argument("--jump-threshold", type=float, default=50.0,
                     help="pnl_pct 급변 플래그 기준(%%p), 기본 50")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"DB 파일을 찾을 수 없음: {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = connect_readonly(db_path)
    try:
        all_keys = list_all_keys(conn)
        if args.strategy:
            if args.mode:
                target_keys = [f"{args.strategy}__{args.mode}"]
            else:
                target_keys = [k for k in all_keys if parse_portfolio_key(k)[0] == args.strategy]
        else:
            target_keys = all_keys

        if not target_keys:
            print("검사할 포트폴리오 키를 찾지 못함 (전략 이름이 맞는지 확인).", file=sys.stderr)
            sys.exit(1)

        for key in target_keys:
            run_for_key(conn, key, args.jump_threshold)

        print(f"\n{'=' * 70}")
        print("참고: 이 스크립트는 '기록된 거래들만으로 지금 저장된 상태(수익률 포함)가 설명되는가'")
        print("즉 거래 기록 ↔ 포트폴리오 문서 간 산술적 일치만 검증한다. 각 거래의 price_krw가")
        print("그 시각 실제 시세 범위 안이었는지(시세 조작/look-ahead bias 여부)는 그 시각의 실제")
        print("분봉 데이터가 있어야 확인 가능 — 필요하면 Upbit 과거 캔들과 대조하는 기능을 추가로 만들 수 있음.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
