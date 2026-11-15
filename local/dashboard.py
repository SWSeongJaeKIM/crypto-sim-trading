"""
로컬 Flask 대시보드. main_loop.py가 쌓는 SQLite(trading.db)를 읍기만 한다
(쓰기 없음, main_loop와 동시에 돌려도 안전).

실행: python3 dashboard.py
접속: 이 PC에서는 http://localhost:5000
      같은 와이파이의 폰/다른 기기에서는 http://<이PC의 사설IP>:5000
      (예: 192.168.0.12:5000 — PC의 IP는 Windows는 ipconfig, Mac은
      `ipconfig getifaddr en0` 로 확인)
"""

from __future__ import annotations

from flask import Flask, jsonify, render_template_string

import storage

app = Flask(__name__)

PAGE = """
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>모의투자 대시보드</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; background:#0f1115; color:#e6e6e6; margin:0; padding:16px; }
  h1 { font-size:20px; margin:0 0 4px; }
  .sub { color:#9aa0a6; font-size:13px; margin-bottom:16px; }
  .cards { display:grid; grid-template-columns: repeat(auto-fit, minmax(220px,1fr)); gap:12px; margin-bottom:20px; }
  .card { background:#1a1d24; border-radius:10px; padding:14px; }
  .card h3 { margin:0 0 8px; font-size:14px; color:#9aa0a6; }
  .pnl-pos { color:#4ade80; font-size:22px; font-weight:700; }
  .pnl-neg { color:#f87171; font-size:22px; font-weight:700; }
  table { width:100%; border-collapse:collapse; font-size:13px; margin-bottom:20px; }
  th, td { text-align:left; padding:6px 8px; border-bottom:1px solid #2a2d35; }
  th { color:#9aa0a6; font-weight:500; }
  .buy { color:#4ade80; } .sell { color:#f87171; }
  .refresh { color:#60a5fa; font-size:12px; cursor:pointer; }
</style>
</head>
<body>
  <h1>🪙 모의투자 대시보드</h1>
  <div class="sub" id="status">불러오는 중...</div>
  <div class="cards" id="cards"></div>
  <h3>최근 거래 20건</h3>
  <table id="trades"><thead><tr><th>시각</th><th>전략</th><th>코인</th><th>구분</th><th>수량</th><th>가격(KRW)</th></tr></thead><tbody></tbody></table>

<script>
async function load() {
  const res = await fetch('/api/data');
  const d = await res.json();
  document.getElementById('status').innerText =
    `마지막 갱신: ${d.meta?.last_run_at || '-'} | 현재가 BTC=${Math.round(d.meta?.prices_krw?.BTC || 0).toLocaleString()}원`;

  const cards = document.getElementById('cards');
  cards.innerHTML = '';
  const sorted = [...d.portfolios].sort((a,b) => b.pnl_pct - a.pnl_pct);
  for (const p of sorted) {
    const cls = p.pnl_pct >= 0 ? 'pnl-pos' : 'pnl-neg';
    cards.innerHTML += `
      <div class="card">
        <h3>${p.strategy}</h3>
        <div class="${cls}">${p.pnl_pct >= 0 ? '+' : ''}${p.pnl_pct.toFixed(2)}%</div>
        <div style="color:#9aa0a6;font-size:12px;margin-top:4px;">평가액 ${Math.round(p.total_value_krw).toLocaleString()}원 · 거래 ${p.trade_count}건</div>
      </div>`;
  }

  const tbody = document.querySelector('#trades tbody');
  tbody.innerHTML = '';
  for (const t of d.trades) {
    tbody.innerHTML += `<tr>
      <td>${t.timestamp.slice(0,19).replace('T',' ')}</td>
      <td>${t.strategy}</td><td>${t.coin}</td>
      <td class="${t.side === 'BUY' ? 'buy' : 'sell'}">${t.side}</td>
      <td>${t.qty.toFixed(4)}</td><td>${Math.round(t.price_krw).toLocaleString()}</td>
    </tr>`;
  }
}
load();
setInterval(load, 15000);
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/api/data")
def api_data():
    with storage.connect() as conn:
        portfolios_raw = storage.get_all_portfolios(conn)
        meta = storage.get_meta(conn, "status") or {}
        prices = meta.get("prices_krw", {})

        portfolios = []
        for name, doc in portfolios_raw.items():
            total = sum(s["cash_krw"] + s["qty"] * prices.get(c, 0) for c, s in doc["slots"].items())
            pnl = (total / doc["initial_capital_krw"] - 1.0) * 100
            portfolios.append({
                "strategy": name,
                "total_value_krw": total,
                "pnl_pct": pnl,
                "trade_count": doc.get("trade_count", 0),
            })

        trades = storage.recent_trades(conn, limit=20)

    return jsonify({"meta": meta, "portfolios": portfolios, "trades": trades})


if __name__ == "__main__":
    print("대시보드 실행: http://localhost:5000  (같은 와이파이의 폰에서는 이 PC의 사설IP:5000)")
    app.run(host="0.0.0.0", port=5000, debug=False)
