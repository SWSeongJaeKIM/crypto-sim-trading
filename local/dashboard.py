"""
로컬 Flask 대시보드. main_loop.py가 쌓는 SQLite(trading.db)를 읍기만 한다
(쓰기 없음, main_loop와 동시에 돌려도 안전).

"AI 분석팀" 탭은 예외적으로 로컬 DB가 아니라 GitHub에서 데이터를 가져온다
— 클라우드에서 1시간마다 도는 6에이전트 코드 분석팀(analysis_team/)이
GitHub에 커밋하는 리포트를 읍어와 보여준다. 인터넷 연결이 없거나 아직
리포트가 한 번도 안 올라왔으면 그냥 빈 상태로 표시된다(에러 아님).

실행: python3 dashboard.py
접속: 이 PC에서는 http://localhost:5000
      같은 와이파이의 폰/다른 기기에서는 http://<이PC의 사설IP>:5000
      (예: 192.168.0.12:5000 — PC의 IP는 Windows는 ipconfig, Mac은
      `ipconfig getifaddr en0` 로 확인)
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from flask import Flask, jsonify, render_template_string

import storage

app = Flask(__name__)

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/SWSeongJaeKIM/crypto-sim-trading/main"
_ANALYSIS_CACHE: dict = {"fetched_at": None, "index": None, "reports": {}}
_ANALYSIS_CACHE_TTL_SEC = 60  # 매 요청마다 GitHub에 때리지 않도록 1분 캐시


def _fetch_github_raw(path: str, timeout: float = 5.0) -> str | None:
    url = f"{GITHUB_RAW_BASE}/{path}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def get_analysis_team_data() -> dict:
    """GitHub의 analysis_team/reports/latest.json + 각 리포트 본문을 가져온다.
    실패하면(오프라인, 아직 리포트 없음 등) 빈 결과를 반환 — 대시보드 전체가
    죽지 않도록 항상 dict를 돌려준다."""
    import time
    now = time.time()
    if _ANALYSIS_CACHE["fetched_at"] and now - _ANALYSIS_CACHE["fetched_at"] < _ANALYSIS_CACHE_TTL_SEC:
        return {"index": _ANALYSIS_CACHE["index"], "reports": _ANALYSIS_CACHE["reports"]}

    index_text = _fetch_github_raw("analysis_team/reports/latest.json")
    if not index_text:
        return {"index": None, "reports": {}}
    try:
        index = json.loads(index_text)
    except json.JSONDecodeError:
        return {"index": None, "reports": {}}

    reports: dict[str, str] = {}
    for role, info in index.get("roles", {}).items():
        path = info.get("path")
        if not path:
            continue
        text = _fetch_github_raw(path)
        if text:
            reports[role] = text

    _ANALYSIS_CACHE["fetched_at"] = now
    _ANALYSIS_CACHE["index"] = index
    _ANALYSIS_CACHE["reports"] = reports
    return {"index": index, "reports": reports}

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
  .tabs { display:flex; gap:8px; margin-bottom:16px; }
  .tab-btn { background:#1a1d24; color:#9aa0a6; border:none; border-radius:8px; padding:8px 16px; font-size:14px; cursor:pointer; }
  .tab-btn.active { background:#2563eb; color:#fff; }
  .tab-page { display:none; }
  .tab-page.active { display:block; }
  .role-card { background:#1a1d24; border-radius:10px; padding:14px; margin-bottom:12px; }
  .role-card h3 { margin:0 0 6px; font-size:15px; }
  .role-summary { color:#d1d5db; font-size:13px; margin-bottom:8px; }
  .role-meta { color:#6b7280; font-size:11px; margin-bottom:8px; }
  .role-toggle { color:#60a5fa; font-size:12px; cursor:pointer; }
  .role-body { white-space:pre-wrap; font-size:13px; color:#d1d5db; margin-top:10px; padding-top:10px; border-top:1px solid #2a2d35; display:none; }
  .role-body.open { display:block; }
  .synth-card { border:1px solid #2563eb; }

  /* 사무실 애니메이션 (AI 분석팀 탭) */
  @keyframes typeBob { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-2px)} }
  @keyframes armType { 0%,100%{transform:rotate(0deg)} 50%{transform:rotate(-8deg)} }
  @keyframes blinkEye { 0%,90%,100%{opacity:1} 95%{opacity:0.1} }
  @keyframes steamRise { 0%{transform:translateY(0) scaleX(1); opacity:0.7} 100%{transform:translateY(-10px) scaleX(1.4); opacity:0} }
  @keyframes deskGlow { 0%,100%{opacity:0.5} 50%{opacity:1} }
  @keyframes badgeBounce { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-4px)} }
  .office-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin-bottom:20px; }
  .desk-card { background:#1a1d24; border-radius:12px; padding:12px 8px 10px; text-align:center; position:relative; overflow:hidden; }
  .desk-card.done { background:#113524; }
  .desk-char { width:60px; height:64px; margin:0 auto 4px; }
  .desk-char .arm-anim { animation:armType 0.5s ease-in-out infinite; }
  .desk-char .head-anim { animation:typeBob 1.1s ease-in-out infinite; }
  .desk-char .eye-anim { animation:blinkEye 3.4s infinite; }
  .desk-steam { animation:steamRise 1.6s ease-out infinite; }
  .desk-role-name { font-size:12px; font-weight:500; color:#e6e6e6; margin:0; }
  .desk-role-state { font-size:11px; color:#9aa0a6; margin:2px 0 0; }
  .desk-role-state.done { color:#4ade80; }
  .desk-badge-done { position:absolute; top:6px; right:6px; width:16px; height:16px; border-radius:50%; background:#22c55e; display:flex; align-items:center; justify-content:center; animation:badgeBounce 1s ease-in-out infinite; font-size:11px; color:#06281a; }
  .desk-lamp { position:absolute; top:4px; left:6px; width:6px; height:6px; border-radius:50%; background:#fbbf24; animation:deskGlow 1.3s ease-in-out infinite; }
</style>
</head>
<body>
  <h1>🪙 모의투자 대시보드</h1>

  <div class="tabs">
    <button class="tab-btn active" id="tab-btn-trading" onclick="showTab('trading')">모의투자</button>
    <button class="tab-btn" id="tab-btn-analysis" onclick="showTab('analysis')">🤖 AI 분석팀</button>
  </div>

  <div class="tab-page active" id="tab-trading">
    <div class="sub" id="status">불러오는 중...</div>
    <div class="progress-bar"><span class="dot dot-idle" id="progress-dot"></span><span id="progress-text">상태 확인 중...</span></div>

    <h3>📅 7일 리셋 포트폴리오 <span style="color:#9aa0a6;font-weight:400;font-size:12px;">(7일마다 1000만원으로 초기화 — 최근 성과 비교용)</span></h3>
    <div class="cards" id="cards-reset"></div>

    <h3>📈 누적 포트폴리오 <span style="color:#9aa0a6;font-weight:400;font-size:12px;">(시작부터 지금까지 쭉 — 장기 성과용)</span></h3>
    <div class="cards" id="cards-cumulative"></div>

    <h3>최근 거래 20건</h3>
    <table id="trades"><thead><tr><th>시각</th><th>전략</th><th>구분</th><th>코인</th><th>매수/매도</th><th>수량</th><th>가격(KRW)</th><th>확신도/베팅비율</th></tr></thead><tbody></tbody></table>

    <h3>LSTM 실전 예측 적중률 (실제 가격으로 확정된 예측 기준)</h3>
    <table id="pred-acc"><thead><tr><th>코인</th><th>확정된 예측 수</th><th>적중</th><th>적중률</th></tr></thead><tbody></tbody></table>

    <h3>LSTM 모델 학습 기록 (코인별 최근 10회)</h3>
    <table id="trainings"><thead><tr><th>시각</th><th>코인</th><th>방식</th><th>데이터기간</th><th>샘플수</th><th>정확도</th></tr></thead><tbody></tbody></table>
  </div>

  <div class="tab-page" id="tab-analysis">
    <div class="sub" id="analysis-status">불러오는 중...</div>
    <div class="office-grid" id="office-grid"></div>
    <div id="analysis-synth"></div>
    <h3>역할별 분석 리포트</h3>
    <div id="analysis-roles"></div>
  </div>

<script>
function showTab(name) {
  document.getElementById('tab-trading').classList.toggle('active', name === 'trading');
  document.getElementById('tab-analysis').classList.toggle('active', name === 'analysis');
  document.getElementById('tab-btn-trading').classList.toggle('active', name === 'trading');
  document.getElementById('tab-btn-analysis').classList.toggle('active', name === 'analysis');
  if (name === 'analysis') loadAnalysis();
}

function toggleRoleBody(id) {
  document.getElementById(id).classList.toggle('open');
}

const ROLE_ORDER = ['strategy', 'lstm', 'execution', 'storage', 'risk', 'synthesizer'];
const ROLE_ICONS = {strategy:'📐', lstm:'🧠', execution:'⚙️', storage:'🗄️', risk:'⚠️', synthesizer:'🏆'};
const ROLE_LABELS_FALLBACK = {strategy:'전략 분석가', lstm:'LSTM 분석가', execution:'실행/버그 분석가',
                                storage:'데이터 분석가', risk:'리스크 분석가', synthesizer:'종합 조정자'};

function deskCharSvg(seed, working) {
  const skin = ['#F4C9A0','#E8A66C','#C98B5E','#8D5A3C'][seed % 4];
  const shirt = ['#7F77DD','#1D9E75','#D85A30','#378ADD','#BA7517','#D4537E'][seed % 6];
  const typing = working ? `<rect x="24" y="44.5" width="12" height="2.6" rx="0.5" fill="#4ade80" opacity="0.8"/>` : '';
  return `
  <svg viewBox="0 0 60 64" width="60" height="64">
    <ellipse cx="30" cy="58" rx="22" ry="4" fill="#2a2d35" opacity="0.4"/>
    <g class="arm-anim" style="transform-origin:16px 44px">
      <rect x="10" y="40" width="10" height="7" rx="3" fill="${shirt}"/>
    </g>
    <g class="arm-anim" style="transform-origin:44px 44px; animation-delay:0.25s">
      <rect x="40" y="40" width="10" height="7" rx="3" fill="${shirt}"/>
    </g>
    <rect x="16" y="32" width="28" height="18" rx="8" fill="${shirt}"/>
    <g class="head-anim">
      <circle cx="30" cy="22" r="14" fill="${skin}"/>
      <path d="M16 18 Q16 6 30 6 Q44 6 44 18 L44 15 Q30 10 16 15 Z" fill="#3b2a1a"/>
      <circle class="eye-anim" cx="25" cy="22" r="1.6" fill="#2c2c2a"/>
      <circle class="eye-anim" cx="35" cy="22" r="1.6" fill="#2c2c2a"/>
      <path d="M26 28 Q30 30 34 28" stroke="#2c2c2a" stroke-width="1.2" fill="none" stroke-linecap="round"/>
    </g>
    <rect x="8" y="48" width="44" height="6" rx="2" fill="#0f1115" stroke="#2a2d35" stroke-width="0.5"/>
    <rect x="22" y="44" width="16" height="4" rx="1" fill="#1a1a1a"/>${typing}
  </svg>`;
}

function renderOffice(roleStates) {
  const grid = document.getElementById('office-grid');
  grid.innerHTML = ROLE_ORDER.map((key, i) => {
    const done = roleStates[key] === 'done';
    const label = ROLE_LABELS_FALLBACK[key];
    return `
    <div class="desk-card ${done ? 'done' : ''}">
      ${!done ? '<div class="desk-lamp"></div>' : '<div class="desk-badge-done">✓</div>'}
      ${!done ? '<div style="position:absolute;top:10px;right:10px;width:4px;height:6px;background:#6b7280;border-radius:2px;opacity:0.5;" class="desk-steam"></div>' : ''}
      <div class="desk-char">${deskCharSvg(i, !done)}</div>
      <p class="desk-role-name">${ROLE_ICONS[key] || ''} ${label}</p>
      <p class="desk-role-state ${done ? 'done' : ''}">${done ? '리포트 완료' : '코드 분석 중...'}</p>
    </div>`;
  }).join('');
}

async function loadAnalysis() {
  const statusEl = document.getElementById('analysis-status');
  statusEl.innerText = '불러오는 중...';
  const res = await fetch('/api/analysis');
  const d = await res.json();

  if (!d.index) {
    statusEl.innerText = '아직 분석 리포트가 없습니다 (클라우드에서 1시간마다 자동 생성되며, 인터넷 연결이 필요합니다). 잠시 후 다시 확인해주세요.';
    const allWorking = {};
    ROLE_ORDER.forEach(k => allWorking[k] = 'working');
    renderOffice(allWorking);
    document.getElementById('analysis-synth').innerHTML = '';
    document.getElementById('analysis-roles').innerHTML = '';
    return;
  }
  statusEl.innerText = `마지막 분석: ${d.index.generated_at || '-'} (UTC) · 1시간마다 자동 갱신`;

  // GitHub에 이번 회차 리포트가 이미 올라온 역할은 "완료" 뱃지, 아직 없는
  // 역할은 "분석 중" 캐릭터로 표시 — latest.json의 roles 키 존재 여부로 판정.
  const roleStates = {};
  ROLE_ORDER.forEach(key => {
    roleStates[key] = d.index.roles?.[key] ? 'done' : 'working';
  });
  renderOffice(roleStates);

  const synthRole = d.index.roles?.synthesizer;
  const synthBody = d.reports?.synthesizer;
  const synthEl = document.getElementById('analysis-synth');
  if (synthRole && synthBody) {
    synthEl.innerHTML = `
      <div class="role-card synth-card">
        <h3>🏆 ${synthRole.role_label} — 이번 회차 종합</h3>
        <div class="role-body open">${synthBody.replace(/</g,'&lt;')}</div>
      </div>`;
  } else {
    synthEl.innerHTML = '';
  }

  const rolesEl = document.getElementById('analysis-roles');
  rolesEl.innerHTML = '';
  for (const roleKey of ROLE_ORDER.filter(k => k !== 'synthesizer')) {
    const info = d.index.roles?.[roleKey];
    if (!info) continue;
    const body = d.reports?.[roleKey] || '(본문을 불러오지 못했습니다)';
    const bodyId = `role-body-${roleKey}`;
    rolesEl.innerHTML += `
      <div class="role-card">
        <h3>${ROLE_ICONS[roleKey] || ''} ${info.role_label}</h3>
        <div class="role-meta">${info.timestamp} (UTC)</div>
        <div class="role-summary">${info.summary}</div>
        <div class="role-toggle" onclick="toggleRoleBody('${bodyId}')">전체 리포트 보기 / 접기</div>
        <div class="role-body" id="${bodyId}">${body.replace(/</g,'&lt;')}</div>
      </div>`;
  }
}
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

  function renderCards(containerId, mode) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    const sorted = d.portfolios.filter(p => p.mode === mode).sort((a,b) => b.pnl_pct - a.pnl_pct);
    for (const p of sorted) {
      const cls = p.pnl_pct >= 0 ? 'pnl-pos' : 'pnl-neg';
      const pnlSign = p.pnl_krw >= 0 ? '+' : '';
      let resetNote = '';
      if (mode === 'reset' && p.created_at) {
        resetNote = `<div style="color:#60a5fa;font-size:11px;margin-top:2px;">이번 주기 시작: ${p.created_at.slice(0,10)}</div>`;
      }
      container.innerHTML += `
        <div class="card">
          <h3>${p.strategy}</h3>
          <div class="${cls}">${p.pnl_pct >= 0 ? '+' : ''}${p.pnl_pct.toFixed(2)}%</div>
          <div class="${cls}" style="font-size:14px;margin-top:2px;">${pnlSign}${Math.round(p.pnl_krw).toLocaleString()}원</div>
          <div style="color:#9aa0a6;font-size:12px;margin-top:4px;">평가액 ${Math.round(p.total_value_krw).toLocaleString()}원 (원금 ${Math.round(p.initial_capital_krw).toLocaleString()}원) · 거래 ${p.trade_count}건</div>
          ${resetNote}
        </div>`;
    }
  }
  renderCards('cards-reset', 'reset');
  renderCards('cards-cumulative', 'cumulative');

  const tbody = document.querySelector('#trades tbody');
  tbody.innerHTML = '';
  for (const t of d.trades) {
    const modeLabel = t.mode === 'reset' ? '리셋' : '누적';
    // reason에 "확신도 0.64 -> 65%" 형태로 박혀있는 부분만 추출해서 별도 칸에 표시
    const m = (t.reason || '').match(/확신도\\s*([\\d.]+)\\s*->\\s*(\\d+)%/);
    const confCell = m ? `${(parseFloat(m[1])*100).toFixed(0)}% → ${m[2]}%` : '-';
    tbody.innerHTML += `<tr>
      <td>${t.timestamp.slice(0,19).replace('T',' ')}</td>
      <td>${t.strategy}</td><td>${modeLabel}</td><td>${t.coin}</td>
      <td class="${t.side === 'BUY' ? 'buy' : 'sell'}">${t.side}</td>
      <td>${t.qty.toFixed(4)}</td><td>${Math.round(t.price_krw).toLocaleString()}</td>
      <td style="color:var(--text-secondary, #9aa0a6);font-size:12px;">${confCell}</td>
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

  const accRows = [...d.prediction_accuracy].sort((a,b) => (b.accuracy ?? -1) - (a.accuracy ?? -1));
  document.querySelector('#pred-acc tbody').innerHTML = accRows.map(a => {
    const accPct = a.accuracy !== null ? (a.accuracy*100).toFixed(1) + '%' : '아직 없음';
    const cls = a.accuracy === null ? '' : (a.accuracy >= 0.5 ? 'acc-up' : 'acc-down');
    return `<tr><td>${a.coin}</td><td>${a.total}</td><td>${a.correct}</td><td class="${cls}">${accPct}</td></tr>`;
  }).join('');
}
load();
setInterval(load, 15000);
// 분석팀 리포트는 1시간마다만 바뀌므로 5분 간격이면 충분 — 지금 분석팀 탭을
// 보고 있을 때만 갱신한다.
setInterval(() => {
  if (document.getElementById('tab-analysis').classList.contains('active')) loadAnalysis();
}, 5 * 60 * 1000);
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/api/analysis")
def api_analysis():
    """클라우드 6에이전트 분석팀이 GitHub에 커밋한 최신 코드 리뷰 리포트.
    로컬 trading.db는 전혀 건드리지 않는다 — GitHub raw 파일만 가져온다."""
    return jsonify(get_analysis_team_data())


@app.route("/api/data")
def api_data():
    with storage.connect() as conn:
        portfolios_raw = storage.get_all_portfolios(conn)
        meta = storage.get_meta(conn, "status") or {}
        progress = storage.get_meta(conn, "progress") or {}
        prices = meta.get("prices_krw", {})

        portfolios = []
        for name, doc in portfolios_raw.items():
            # name은 "<strategy>__reset" 또는 "<strategy>__cumulative" 형태.
            strategy, _, mode = name.rpartition("__")
            if mode not in ("reset", "cumulative"):
                strategy, mode = name, "cumulative"  # 혹시 옛 데이터가 남아있으면 누적으로 간주
            total = sum(s["cash_krw"] + s["qty"] * prices.get(c, 0) for c, s in doc["slots"].items())
            initial = doc["initial_capital_krw"]
            pnl = (total / initial - 1.0) * 100
            portfolios.append({
                "strategy": strategy,
                "mode": mode,
                "total_value_krw": total,
                "initial_capital_krw": initial,
                "pnl_krw": total - initial,
                "pnl_pct": pnl,
                "trade_count": doc.get("trade_count", 0),
                "created_at": doc.get("created_at"),
            })

        trades = storage.recent_trades(conn, limit=20)
        for t in trades:
            strategy, _, mode = t["strategy"].rpartition("__")
            if mode in ("reset", "cumulative"):
                t["strategy"] = strategy
                t["mode"] = mode
            else:
                t["mode"] = "cumulative"

        # 코인별 최근 10회, 화면엔 시간순(과거->최신)으로 묶어서 내려줘야
        # 프론트에서 "이전 대비 ▲▼" 추이 계산이 바로 된다.
        trainings: list[dict] = []
        for coin in ["BTC", "ETH", "XRP", "SOL", "DOGE", "ADA"]:
            rows = storage.model_training_history(conn, coin=coin, limit=10)
            trainings.extend(reversed(rows))  # 오래된 것 먼저

        prediction_accuracy = storage.prediction_accuracy_by_coin(conn)

    return jsonify({"meta": meta, "progress": progress, "portfolios": portfolios,
                     "trades": trades, "trainings": trainings,
                     "prediction_accuracy": prediction_accuracy})


if __name__ == "__main__":
    print("대시보드 실행: http://localhost:5000  (같은 와이파이의 폰에서는 이 PC의 사설IP:5000)")
    app.run(host="0.0.0.0", port=5000, debug=False)
