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
  .progress-bar { background:#1a1d24; border-radius:10px; padding:12px 14px; margin-bottom:16px; display:flex; align-items:center; gap:10px; }
  .dot { width:10px; height:10px; border-radius:50%; flex-shrink:0; }
  .dot-idle { background:#6b7280; } .dot-collecting,.dot-cycle_running { background:#60a5fa; animation:pulse 1.5s infinite; }
  .dot-training { background:#fbbf24; animation:pulse 1.5s infinite; } .dot-error { background:#f87171; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
  .acc-up { color:#4ade80; } .acc-down { color:#f87171; } .acc-flat { color:#9aa0a6; }
</style>
</head>
<body>
  <h1>🪙 모의투자 대시보드</h1>
  <div class="sub" id="status">불러오는 중...</div>
  <div class="progress-bar"><span class="dot dot-idle" id="progress-dot"></span><span id="progress-text">상태 확인 중...</span></div>
  <div class="cards" id="cards"></div>
  <h3>최근 거래 20건</h3>
  <table id="trades"><thead><tr><th>시각</th><th>전략</th><th>코인</th><th>구분</th><th>수량</th><th>가격(KRW)</th></tr></thead><tbody></tbody></table>

  <h3>LSTM 모델 학습 기록 (코인별 최근 10회)</h3>
  <table id="trainings"><thead><tr><th>시각</th><th>코인</th><th>방식</th><th>데이터기간</th><th>샘플수</th><th>정확도</th></tr></thead><tbody></tbody></table>

<script>
async function load() {
  const res = await fetch('/api/data');
  const d = await res.json();
  document.getElementById('status').innerText =
    `마지막 갱신: ${d.meta?.last_run_at || '-'} | 현재가 BTC=${Math.round(d.meta?.prices_krw?.BTC || 0).toLocaleString()}원`;

  const p = d.progress || {};
  const dot = document.getElementById('progress-dot');
  dot.className = 'dot dot-' + (p.stage || 'idle');
  const stageLabel = {idle:'대기 중', collecting:'데이터 수집 중', cycle_running:'매매 사이클 실행 중',
                      training:'LSTM 학습 중', sleeping:'대기 중', error:'에러'}[p.stage] || p.stage || '알수없음';
  document.getElementById('progress-text').innerText =
    `[${stageLabel}] ${p.detail || ''}` + (p.updated_at ? ` (${p.updated_at.slice(11,19)})` : '');

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

  // d.trainings는 코인별로 시간순(과거->최신) 묶음 — 먼저 추이(▲▼)를 계산한 뒤
  // 화면엔 최신이 위로 오도록 뒤집어서 보여준다.
  const lastAccByCoin = {};
  const rowsHtml = [];
  for (const t of d.trainings) {
    const prev = lastAccByCoin[t.coin];
    lastAccByCoin[t.coin] = t.val_accuracy;
    let trend = '';
    if (prev !== undefined && t.val_accuracy !== null) {
      if (t.val_accuracy > prev) trend = '<span class="acc-up">▲</span>';
      else if (t.val_accuracy < prev) trend = '<span class="acc-down">▼</span>';
      else trend = '<span class="acc-flat">–</span>';
    }
    const acc = t.val_accuracy !== null ? (t.val_accuracy*100).toFixed(1) + '%' : '-';
    const period = (t.data_start && t.data_end) ? `${t.data_start.slice(0,10)}~${t.data_end.slice(0,10)}` : '-';
    rowsHtml.push(`<tr>
      <td>${(t.timestamp||'').slice(0,19).replace('T',' ')}</td>
      <td>${t.coin}</td>
      <td>${t.model_type === 'lstm_finetune' ? '추가학습' : '최초학습'}</td>
      <td>${period}</td>
      <td>${t.n_samples ?? '-'}</td>
      <td>${acc} ${trend}</td>
    </tr>`);
  }
  document.querySelector('#trainings tbody').innerHTML = rowsHtml.reverse().join('');
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
        progress = storage.get_meta(conn, "progress") or {}
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

        # 코인별 최근 10회, 화면엔 시간순(과거->최신)으로 묶어서 내려줘야
        # 프론트에서 "이전 대비 ▲▼" 추이 계산이 바로 된다.
        trainings: list[dict] = []
        for coin in ["BTC", "ETH", "XRP", "SOL", "DOGE", "ADA"]:
            rows = storage.model_training_history(conn, coin=coin, limit=10)
            trainings.extend(reversed(rows))  # 오래된 것 먼저

    return jsonify({"meta": meta, "progress": progress, "portfolios": portfolios,
                     "trades": trades, "trainings": trainings})


if __name__ == "__main__":
    print("대시보드 실행: http://localhost:5000  (같은 와이파이의 폰에서는 이 PC의 사설IP:5000)")
    app.run(host="0.0.0.0", port=5000, debug=False)
