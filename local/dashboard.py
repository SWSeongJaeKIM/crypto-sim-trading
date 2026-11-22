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
import re
import urllib.error
import urllib.request

from flask import Flask, jsonify, render_template_string

import lstm_model
import portfolio
import storage
import verify_ledger

app = Flask(__name__)

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/SWSeongJaeKIM/crypto-sim-trading/main"
GITHUB_API_CONTENTS = "https://api.github.com/repos/SWSeongJaeKIM/crypto-sim-trading/contents/analysis_team/reports"
_ANALYSIS_CACHE: dict = {"fetched_at": None, "index": None, "reports": {}}
_ANALYSIS_CACHE_TTL_SEC = 60  # 매 요청마다 GitHub에 때리지 않도록 1분 캐시
_HISTORY_CACHE: dict = {"fetched_at": None, "timestamps": []}
_HISTORY_CACHE_TTL_SEC = 300  # 회차 목록은 5분에 한 번만 다시 조회


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


def get_analysis_history_timestamps() -> list[str]:
    """GitHub에 쌓인 모든 회차의 타임스탬프 목록(최신순)을 돌려준다.
    파일명 규칙(reports/<YYYY-MM-DD_HHMM>_<role>.md)에서 타임스탬프만 추출해
    중복 제거. GitHub Contents API를 쓰므로 비인증 호출 레이트리밋(시간당
    60회)에 걸릴 수 있는데, 실패하면 빈 리스트를 반환해 조용히 넘어간다."""
    import time
    now = time.time()
    if _HISTORY_CACHE["fetched_at"] and now - _HISTORY_CACHE["fetched_at"] < _HISTORY_CACHE_TTL_SEC:
        return _HISTORY_CACHE["timestamps"]

    try:
        req = urllib.request.Request(
            GITHUB_API_CONTENTS, headers={"Accept": "application/vnd.github+json"}
        )
        with urllib.request.urlopen(req, timeout=6.0) as resp:
            listing = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return _HISTORY_CACHE["timestamps"]  # 실패하면 이전 캐시(없으면 빈 리스트) 유지

    timestamps = set()
    for entry in listing if isinstance(listing, list) else []:
        name = entry.get("name", "")
        m = re.match(r"^(\d{4}-\d{2}-\d{2}_\d{4})_[a-z_]+\.md$", name)
        if m:
            timestamps.add(m.group(1))

    sorted_ts = sorted(timestamps, reverse=True)  # 최신 회차 먼저
    _HISTORY_CACHE["fetched_at"] = now
    _HISTORY_CACHE["timestamps"] = sorted_ts
    return sorted_ts


def get_analysis_round(timestamp: str) -> dict:
    """특정 회차(timestamp)의 6개 역할 리포트 본문을 GitHub에서 가져온다."""
    reports: dict[str, str] = {}
    for role in ["strategy", "lstm", "execution", "storage", "risk", "synthesizer"]:
        text = _fetch_github_raw(f"analysis_team/reports/{timestamp}_{role}.md")
        if text:
            reports[role] = text
    return reports

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
  .short-open { color:#fb923c; } .short-close { color:#60a5fa; } .liquidated { color:#f87171; font-weight:bold; }
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
    <button class="tab-btn" id="tab-btn-verify" onclick="showTab('verify')">🔍 원장 검증</button>
  </div>

  <div class="tab-page active" id="tab-trading">
    <div class="sub" id="status">불러오는 중...</div>
    <div class="progress-bar"><span class="dot dot-idle" id="progress-dot"></span><span id="progress-text">상태 확인 중...</span></div>

    <h3>📅 7일 리셋 포트폴리오 <span style="color:#9aa0a6;font-weight:400;font-size:12px;">(7일마다 1000만원으로 초기화 — 최근 성과 비교용, 카드를 누르면 코인별 보유 현황이 펼쳐집니다)</span></h3>
    <div class="cards" id="cards-reset"></div>

    <h3>📈 누적 포트폴리오 <span style="color:#9aa0a6;font-weight:400;font-size:12px;">(시작부터 지금까지 쭉 — 장기 성과용, 카드를 누르면 코인별 보유 현황이 펼쳐집니다)</span></h3>
    <div class="cards" id="cards-cumulative"></div>

    <h3>거래 내역 <span id="trades-count" style="color:#9aa0a6;font-weight:400;font-size:12px;"></span></h3>
    <div style="display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin-bottom:10px;">
      <label style="font-size:12px; color:#9aa0a6;">전략 <select id="filter-strategy" onchange="renderTrades()" style="background:#1a1d24; color:#e6e6e6; border:1px solid #2a2d35; border-radius:6px; padding:4px 8px; font-size:12px;"><option value="">전체</option></select></label>
      <label style="font-size:12px; color:#9aa0a6;">구분 <select id="filter-mode" onchange="renderTrades()" style="background:#1a1d24; color:#e6e6e6; border:1px solid #2a2d35; border-radius:6px; padding:4px 8px; font-size:12px;"><option value="">전체</option><option value="reset">리셋</option><option value="cumulative">누적</option></select></label>
      <label style="font-size:12px; color:#9aa0a6;">코인 <select id="filter-coin" onchange="renderTrades()" style="background:#1a1d24; color:#e6e6e6; border:1px solid #2a2d35; border-radius:6px; padding:4px 8px; font-size:12px;"><option value="">전체</option></select></label>
      <label style="font-size:12px; color:#9aa0a6;">매수/매도 <select id="filter-side" onchange="renderTrades()" style="background:#1a1d24; color:#e6e6e6; border:1px solid #2a2d35; border-radius:6px; padding:4px 8px; font-size:12px;">
        <option value="">전체</option>
        <option value="BUY">매수</option>
        <option value="SELL">매도</option>
        <option value="SHORT_OPEN">숏 진입</option>
        <option value="SHORT_CLOSE">숏 청산</option>
        <option value="LONG_LIQUIDATED">롱 강제청산</option>
        <option value="SHORT_LIQUIDATED">숏 강제청산</option>
      </select></label>
      <button onclick="resetTradeFilters()" style="background:#1a1d24; color:#9aa0a6; border:1px solid #2a2d35; border-radius:6px; padding:4px 10px; font-size:12px; cursor:pointer;">필터 초기화</button>
    </div>

    <table id="trades"><thead><tr><th>시각</th><th>전략</th><th>구분</th><th>코인</th><th>매수/매도</th><th>수량</th><th>가격(KRW)</th><th>확신도/베팅비율</th></tr></thead><tbody></tbody></table>

    <h3>LSTM 실전 예측 적중률 (실제 가격으로 확정된 예측 기준)</h3>
    <table id="pred-acc"><thead><tr><th>코인</th><th>확정된 예측 수</th><th>적중</th><th>적중률</th></tr></thead><tbody></tbody></table>

    <h3>코인별 손절/강제청산 발생 횟수 <span style="color:#9aa0a6;font-weight:400;font-size:12px;">(전체 기간 누적 — 변동성이 큰 코인이 실제로 더 자주 손절에 걸리는지 참고용. 이 집계만으로 손절선/배분을 바로 바꾸지는 않음)</span></h3>
    <table id="stop-loss-counts"><thead><tr><th>코인</th><th>손절</th><th>강제청산</th><th>합계</th></tr></thead><tbody></tbody></table>

    <h3>LSTM 모델 학습 기록 (코인별 최근 10회)</h3>
    <table id="trainings"><thead><tr><th>시각</th><th>코인</th><th>방식</th><th>데이터기간</th><th>샘플수</th><th>정확도</th></tr></thead><tbody></tbody></table>
  </div>

  <div class="tab-page" id="tab-analysis">
    <div class="sub" id="analysis-status">불러오는 중...</div>
    <div class="office-grid" id="office-grid"></div>

    <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">
      <label for="round-select" style="font-size:13px; color:#9aa0a6;">회차 선택:</label>
      <select id="round-select" onchange="onRoundChange()" style="background:#1a1d24; color:#e6e6e6; border:1px solid #2a2d35; border-radius:6px; padding:6px 10px; font-size:13px;"></select>
    </div>

    <div id="analysis-synth"></div>
    <h3>역할별 분석 리포트 (에이전트들이 한 말 전체)</h3>
    <div id="analysis-roles"></div>
  </div>

  <div class="tab-page" id="tab-verify">
    <div class="sub" id="verify-status">불러오는 중...</div>

    <h3>🧠 자동매매 AI(LSTM) 모델 구조</h3>
    <div class="card" id="model-info-card" style="margin-bottom:20px;"></div>

    <h3>거래 원장 ↔ 포트폴리오 정합성 검증 <span style="color:#9aa0a6;font-weight:400;font-size:12px;">(기록된 거래들만으로 지금 저장된 cash/보유량/수익률이 전부 설명되는지 재계산해서 대조 — "이 전략의 수익이 말이 안 된다" 싶을 때 가장 먼저 볼 곳)</span></h3>
    <div id="verify-results"></div>
  </div>

<script>
function showTab(name) {
  document.getElementById('tab-trading').classList.toggle('active', name === 'trading');
  document.getElementById('tab-analysis').classList.toggle('active', name === 'analysis');
  document.getElementById('tab-verify').classList.toggle('active', name === 'verify');
  document.getElementById('tab-btn-trading').classList.toggle('active', name === 'trading');
  document.getElementById('tab-btn-analysis').classList.toggle('active', name === 'analysis');
  document.getElementById('tab-btn-verify').classList.toggle('active', name === 'verify');
  if (name === 'analysis') loadAnalysis();
  if (name === 'verify') loadVerify();
}

let lastTradesData = null;  // 마지막으로 받은 d.trades — 필터 select를 바꿀 때 재요청 없이 다시 그리는 데 사용
let lastFullData = null;    // 마지막으로 받은 전체 응답(d) — 카드를 펼치고 접을 때 재요청 없이 다시 그리는 데 사용
let lastProgress = null;    // 마지막으로 받은 progress — "N초 전"을 15초 서버 폴링과 별개로 매초 갱신하는 데 사용

// "마지막 업데이트 Ν초 전" 표시를 서버에서 새 데이터를 받아올 때만(15초마다)
// 갱신하면 화면에서 숫자가 15초 단위로 뚝뚝 끊겨 보인다 — updated_at 자체는
// 그대로 두고, 지금 몇 초가 지났는지 계산하는 쪽을 1초 간격으로 따로 돌린다.
function tickProgressDisplay() {
  const p = lastProgress;
  if (!p) return;
  const dot = document.getElementById('progress-dot');
  const stageLabel = {idle:'대기 중', collecting:'데이터 수집 중', cycle_running:'매매 사이클 실행 중',
                      training:'LSTM 학습 중', sleeping:'대기 중', error:'에러'}[p.stage] || p.stage || '알수없음';
  // progress.updated_at은 main_loop.py가 set_progress()를 호출할 때마다
  // 찍히는 시각 — watchdog.bat가 보는 heartbeat.txt와 같은 순간에 갱신되는
  // 값이라, 여기서 "몇 분 전"을 보여주면 워치독이 멈춤으로 판단하기 전에
  // 사람도 눈으로 먼저 알아챌 수 있다.
  let ageLabel = '';
  let staleWarning = false;
  if (p.updated_at) {
    const ageSec = (Date.now() - new Date(p.updated_at).getTime()) / 1000;
    if (ageSec < 90) {
      ageLabel = `${Math.round(ageSec)}초 전`;
    } else {
      const ageMin = Math.round(ageSec / 60);
      ageLabel = `${ageMin}분 전`;
      if (ageMin >= 5) staleWarning = true;  // watchdog.bat의 STALE_MINUTES(기본 5분)와 동일 기준
    }
  }
  dot.className = 'dot dot-' + (staleWarning ? 'error' : (p.stage || 'idle'));
  document.getElementById('progress-text').innerText =
    `[${stageLabel}] ${p.detail || ''}` + (ageLabel ? ` (${ageLabel})` : '') +
    (staleWarning ? ' ⚠ 응답 없음 — 멈췄을 수 있음 (watchdog 사용 시 자동 재시작됨)' : '');
}
const expandedCards = new Set();  // 펼쳐진 카드 키("<mode>__<strategy>") 집합. 15초마다 새로고침해도 펼친 상태가 유지되게 한다.

function toggleCardDetail(cardKey) {
  if (expandedCards.has(cardKey)) expandedCards.delete(cardKey);
  else expandedCards.add(cardKey);
  if (!lastFullData) return;
  renderCards('cards-reset', 'reset');
  renderCards('cards-cumulative', 'cumulative');
}

// 전략 카드를 펼쳤을 때 보여줄 "코인별 보유 현황". 처음엔 6개 컬럼짜리
// <table>이었는데, 카드 하나 너비(grid auto-fit라 보통 220~300px)에 6개
// 컬럼을 욱여넣다 보니 "레버리지" 같은 한글이 세로로 한 글자씩 줄바꿈되는
// 등 전혀 못 읽을 수준이 됐었다(실제 스크린샷으로 확인). 두 가지로 고쳤다:
// 1) renderCards에서 펼쳐진 카드는 grid-column을 전체 폭으로 넓힌다.
// 2) 표 대신, 코인 하나당 한 줄(flex row)로 자연스럽게 흐르는 리스트로 바꿔서
//    좁은 화면에서도 글자가 줄 단위로만 줄바꿈되지 하나씩 쪼개지지 않게 한다.
// 또한 "대기 중"(포지션 없음) 코인을 한 줄씩 다 나열하지 않고 한 줄로 묶어서,
// 정작 필요한 "지금 들어가 있는 포지션"만 바로 눈에 띄게 했다.
function renderPositionDetail(slots) {
  if (!slots.length) return '';
  const active = slots.filter(s => s.position !== 'none');
  const idle = slots.filter(s => s.position === 'none');

  const rows = active.map(s => {
    const isShort = s.position === 'short';
    const entryPrice = isShort ? s.short_avg_entry_krw : s.avg_cost_krw;
    const badgeLabel = isShort ? `숏 ${s.leverage.toFixed(0)}배` : '롱';
    const badgeColor = isShort ? '#eda100' : '#3987e5';
    const priceLabel = s.current_price_krw ? `${Math.round(s.current_price_krw).toLocaleString()}원` : '-';
    const entryLabel = entryPrice ? `${Math.round(entryPrice).toLocaleString()}원` : '-';

    let pnlLabel = '-';
    let pnlCls = '';
    if (s.unrealized_pnl_pct !== null && s.unrealized_pnl_pct !== undefined) {
      pnlCls = s.unrealized_pnl_pct >= 0 ? 'pnl-pos' : 'pnl-neg';
      const sign = s.unrealized_pnl_pct >= 0 ? '+' : '';
      pnlLabel = `${sign}${s.unrealized_pnl_pct.toFixed(1)}% (${sign}${Math.round(s.unrealized_pnl_krw).toLocaleString()}원)`;
    }

    return `
      <div style="display:flex; flex-wrap:wrap; align-items:center; gap:8px; padding:7px 0; border-bottom:1px solid #22252c; font-size:13px;">
        <span style="font-weight:700; width:42px; flex-shrink:0;">${s.coin}</span>
        <span style="background:${badgeColor}22; color:${badgeColor}; border-radius:5px; padding:2px 7px; font-size:11px; font-weight:600; flex-shrink:0;">${badgeLabel}</span>
        <span style="color:#9aa0a6; white-space:nowrap;">${entryLabel} → ${priceLabel}</span>
        <span class="${pnlCls}" style="margin-left:auto; font-weight:700; white-space:nowrap;">${pnlLabel}</span>
      </div>`;
  }).join('');

  const idleLine = idle.length
    ? `<div style="color:#5b6472; font-size:12px; margin-top:6px;">대기 중(현금 보유): ${idle.map(s => s.coin).join(', ')}</div>`
    : '';
  const emptyLine = active.length
    ? ''
    : `<div style="color:#5b6472; font-size:12px; padding:4px 0;">현재 보유 중인 포지션이 없습니다 — 전액 현금 대기 중</div>`;

  return `
    <div style="margin-top:10px; border-top:1px solid #2a2d35; padding-top:8px;" onclick="event.stopPropagation();">
      ${rows}${emptyLine}${idleLine}
    </div>`;
}

function renderCards(containerId, mode) {
  if (!lastFullData) return;
  const container = document.getElementById(containerId);
  container.innerHTML = '';
  const sorted = lastFullData.portfolios.filter(p => p.mode === mode).sort((a,b) => b.pnl_pct - a.pnl_pct);
  for (const p of sorted) {
    const cls = p.pnl_pct >= 0 ? 'pnl-pos' : 'pnl-neg';
    const pnlSign = p.pnl_krw >= 0 ? '+' : '';
    let resetNote = '';
    if (mode === 'reset' && p.created_at) {
      resetNote = `<div style="color:#60a5fa;font-size:11px;margin-top:2px;">이번 주기 시작: ${p.created_at.slice(0,10)}</div>`;
    }
    const cardKey = `${mode}__${p.strategy}`;
    const isOpen = expandedCards.has(cardKey);
    container.innerHTML += `
      <div class="card" onclick="toggleCardDetail('${cardKey}')" style="cursor:pointer;${isOpen ? ' grid-column: 1 / -1;' : ''}">
        <h3>${p.strategy} <span style="float:right; color:#5b6472; font-size:11px;">${isOpen ? '▲ 접기' : '▼ 코인별 현황'}</span></h3>
        <div class="${cls}">${p.pnl_pct >= 0 ? '+' : ''}${p.pnl_pct.toFixed(2)}%</div>
        <div class="${cls}" style="font-size:14px;margin-top:2px;">${pnlSign}${Math.round(p.pnl_krw).toLocaleString()}원</div>
        <div style="color:#9aa0a6;font-size:12px;margin-top:4px;">평가액 ${Math.round(p.total_value_krw).toLocaleString()}원 (원금 ${Math.round(p.initial_capital_krw).toLocaleString()}원) · 거래 ${p.trade_count}건</div>
        <div style="color:#9aa0a6;font-size:11px;margin-top:2px;">최대낙폭(MDD) -${(p.max_drawdown_pct || 0).toFixed(2)}%</div>
        ${resetNote}
        <div style="display:${isOpen ? 'block' : 'none'};">${renderPositionDetail(p.slots || [])}</div>
      </div>`;
  }
}

function resetTradeFilters() {
  document.getElementById('filter-strategy').value = '';
  document.getElementById('filter-mode').value = '';
  document.getElementById('filter-coin').value = '';
  document.getElementById('filter-side').value = '';
  renderTrades();
}

function renderTrades() {
  if (!lastTradesData) return;
  const fStrategy = document.getElementById('filter-strategy').value;
  const fMode = document.getElementById('filter-mode').value;
  const fCoin = document.getElementById('filter-coin').value;
  const fSide = document.getElementById('filter-side').value;

  // select 옵션(전략/코인)은 매번 받아온 데이터 기준으로 다시 채운다 — 선택값은 유지.
  function fillOptions(selectId, values) {
    const sel = document.getElementById(selectId);
    const current = sel.value;
    const opts = ['<option value="">전체</option>']
      .concat(values.map(v => `<option value="${v}">${v}</option>`));
    sel.innerHTML = opts.join('');
    if (values.includes(current)) sel.value = current;
  }
  fillOptions('filter-strategy', [...new Set(lastTradesData.map(t => t.strategy))].sort());
  fillOptions('filter-coin', [...new Set(lastTradesData.map(t => t.coin))].sort());

  const filtered = lastTradesData.filter(t =>
    (!fStrategy || t.strategy === fStrategy) &&
    (!fMode || t.mode === fMode) &&
    (!fCoin || t.coin === fCoin) &&
    (!fSide || t.side === fSide)
  );

  document.getElementById('trades-count').innerText =
    `(최근 ${lastTradesData.length}건 중 ${filtered.length}건 표시)`;

  const tbody = document.querySelector('#trades tbody');
  tbody.innerHTML = '';
  for (const t of filtered) {
    const modeLabel = t.mode === 'reset' ? '리셋' : '누적';
    // reason에 "확신도 0.64 -> 65%" 형태로 박혀있는 부분만 추출해서 별도 칸에 표시
    const m = (t.reason || '').match(/확신도\\s*([\\d.]+)\\s*->\\s*(\\d+)%/);
    const confCell = m ? `${(parseFloat(m[1])*100).toFixed(0)}% → ${m[2]}%` : '-';
    const sideMeta = {
      BUY: ['buy', '매수'],
      SELL: ['sell', '매도'],
      SHORT_OPEN: ['short-open', '숏 진입'],
      SHORT_CLOSE: ['short-close', '숏 청산'],
      LONG_LIQUIDATED: ['liquidated', '롱 강제청산'],
      SHORT_LIQUIDATED: ['liquidated', '숏 강제청산'],
    };
    const [sideClass, sideLabel] = sideMeta[t.side] || ['sell', t.side];
    tbody.innerHTML += `<tr>
      <td>${t.timestamp.slice(0,19).replace('T',' ')}</td>
      <td>${t.strategy}</td><td>${modeLabel}</td><td>${t.coin}</td>
      <td class="${sideClass}">${sideLabel}</td>
      <td>${t.qty.toFixed(4)}</td><td>${Math.round(t.price_krw).toLocaleString()}</td>
      <td style="color:var(--text-secondary, #9aa0a6);font-size:12px;">${confCell}</td>
    </tr>`;
  }
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

// roleInfoMap: {role: {role_label, timestamp, summary}} (latest.json 형태, 없으면 null)
// reports: {role: '본문 텍스트'}
// 리포트 본문은 항상 펼쳐진 채로 보여준다 — "에이전트들이 한 말"을 바로 읽을 수 있게.
function renderRound(reports, roleInfoMap) {
  const synthEl = document.getElementById('analysis-synth');
  const synthBody = reports?.synthesizer;
  if (synthBody) {
    const synthLabel = roleInfoMap?.synthesizer?.role_label || ROLE_LABELS_FALLBACK.synthesizer;
    synthEl.innerHTML = `
      <div class="role-card synth-card">
        <h3>🏆 ${synthLabel} — 이 회차 종합</h3>
        <div class="role-body open">${synthBody.replace(/</g,'&lt;')}</div>
      </div>`;
  } else {
    synthEl.innerHTML = '';
  }

  const rolesEl = document.getElementById('analysis-roles');
  rolesEl.innerHTML = '';
  for (const roleKey of ROLE_ORDER.filter(k => k !== 'synthesizer')) {
    const body = reports?.[roleKey];
    if (!body) continue;
    const info = roleInfoMap?.[roleKey];
    const label = info?.role_label || ROLE_LABELS_FALLBACK[roleKey];
    const meta = info?.timestamp ? `<div class="role-meta">${info.timestamp} (UTC)</div>` : '';
    const summary = info?.summary ? `<div class="role-summary">${info.summary}</div>` : '';
    rolesEl.innerHTML += `
      <div class="role-card">
        <h3>${ROLE_ICONS[roleKey] || ''} ${label}</h3>
        ${meta}${summary}
        <div class="role-body open">${body.replace(/</g,'&lt;')}</div>
      </div>`;
  }
}

async function onRoundChange() {
  const sel = document.getElementById('round-select');
  const ts = sel.value;
  if (!ts) return;
  if (ts === '__latest__') {
    const res = await fetch('/api/analysis');
    const d = await res.json();
    renderRound(d.reports, d.index?.roles);
    return;
  }
  const res = await fetch(`/api/analysis/round/${ts}`);
  const d = await res.json();
  renderRound(d.reports, null);
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

  // 회차 선택 드롭다운: 최신 회차를 기본값으로, 과거 회차는 이력 API에서 받아온다.
  const sel = document.getElementById('round-select');
  const wasEmpty = sel.options.length === 0;
  if (wasEmpty) {
    sel.innerHTML = `<option value="__latest__">최신 (${d.index.generated_at || '-'})</option>`;
    try {
      const histRes = await fetch('/api/analysis/history');
      const hist = await histRes.json();
      for (const ts of (hist.timestamps || [])) {
        if (ts === d.index.generated_at) continue;  // 최신은 이미 위에 있음
        const opt = document.createElement('option');
        opt.value = ts;
        opt.textContent = ts;
        sel.appendChild(opt);
      }
    } catch (e) { /* 이력 조회 실패해도 최신 회차는 계속 보여줌 */ }
  }

  // 지금 "최신"이 선택된 상태일 때만 자동 갱신 내용을 반영 (과거 회차를 보고 있으면 안 덮어씀)
  if (sel.value === '__latest__' || wasEmpty) {
    sel.value = '__latest__';
    renderRound(d.reports, d.index.roles);
  }
}
const VERIFY_LABELS = {
  duplicates: '중복 거래',
  qty_anomalies: '보유량 초과 매도/청산',
  reconciliation_issues: '최종 상태 대조 불일치',
  stop_loss_issues: '손절/강제청산 reason 임계값 불일치',
};

function renderVerifyIssueList(items, label) {
  if (!items || !items.length) return '';
  const rows = items.slice(0, 20).map(it =>
    `<div style="padding:4px 0; border-bottom:1px solid #22252c; font-family:monospace; font-size:12px; color:#d1d5db;">${JSON.stringify(it)}</div>`
  ).join('');
  return `<div style="margin-top:10px;"><div style="color:#f87171; font-size:13px; font-weight:600;">⚠ ${label} (${items.length}건)</div>${rows}</div>`;
}

function renderVerify(data) {
  const mi = data.model_info || {};
  document.getElementById('model-info-card').innerHTML = `
    <div style="color:#d1d5db; font-size:13px; line-height:1.8;">
      종류: ${mi.architecture || '-'} (hidden=${mi.hidden_size ?? '-'}, layers=${mi.num_layers ?? '-'})<br>
      입력 시퀀스 길이: 과거 ${mi.seq_len_minutes ?? '-'}분<br>
      입력 피처(${mi.n_features ?? '-'}개): ${(mi.feature_cols || []).join(', ')}<br>
      <span style="color:#6b7280; font-size:12px;">※ 코드에 고정된 아키텍처 정보. 실제 학습 시각/정확도 추이는 위 "모의투자" 탭의 "LSTM 모델 학습 기록" 표 참고.</span>
    </div>`;

  const resultsEl = document.getElementById('verify-results');
  resultsEl.innerHTML = '';
  for (const r of (data.results || [])) {
    if (!r.found) {
      resultsEl.innerHTML += `<div class="card" style="margin-bottom:10px;"><h3>${r.key}</h3><div style="color:#9aa0a6;">${r.note}</div></div>`;
      continue;
    }
    const badgeHtml = r.ok
      ? `<span class="pnl-pos" style="font-size:13px;">✓ 이상 없음</span>`
      : `<span class="pnl-neg" style="font-size:13px;">⚠ 확인 필요</span>`;
    let issuesHtml = '';
    for (const field of ['duplicates', 'qty_anomalies', 'reconciliation_issues', 'stop_loss_issues']) {
      issuesHtml += renderVerifyIssueList(r[field], VERIFY_LABELS[field]);
    }
    if (r.pnl_jump_flags && r.pnl_jump_flags.length) {
      issuesHtml += renderVerifyIssueList(r.pnl_jump_flags, '수익률 급변 구간(버그 확정 아님, 참고용)');
    }
    resultsEl.innerHTML += `
      <div class="card" style="margin-bottom:10px;">
        <h3>${r.strategy} <span style="color:#9aa0a6; font-weight:400; font-size:12px;">(${r.variant === 'reset' ? '리셋' : '누적'})</span> ${badgeHtml}</h3>
        <div style="color:#9aa0a6; font-size:12px;">${r.scope_note}</div>
        ${issuesHtml}
      </div>`;
  }
  if (!(data.results || []).length) {
    resultsEl.innerHTML = '<div style="color:#9aa0a6;">검증할 포트폴리오가 아직 없습니다.</div>';
  }
}

async function loadVerify() {
  const statusEl = document.getElementById('verify-status');
  statusEl.innerText = '검증 중...';
  try {
    const res = await fetch('/api/verify');
    const d = await res.json();
    renderVerify(d);
    const okCount = (d.results || []).filter(r => r.found && r.ok).length;
    const totalCount = (d.results || []).filter(r => r.found).length;
    statusEl.innerText = `${totalCount}개 전략×모드 중 ${okCount}개 정상 — 거래 기록과 포트폴리오 문서 대조 기준 `
      + `(가격이 그 시각 실제 시세였는지는 별도 확인 필요)`;
  } catch (e) {
    statusEl.innerText = '검증 중 오류가 발생했습니다.';
  }
}

async function load() {
  const res = await fetch('/api/data');
  const d = await res.json();
  const dbSizeLabel = d.db_size_mb !== undefined ? ` | DB 크기: ${d.db_size_mb}MB` : '';
  document.getElementById('status').innerText =
    `마지막 갱신: ${d.meta?.last_run_at || '-'} | 현재가 BTC=${Math.round(d.meta?.prices_krw?.BTC || 0).toLocaleString()}원${dbSizeLabel}`;

  lastProgress = d.progress || {};
  tickProgressDisplay();

  lastFullData = d;
  renderCards('cards-reset', 'reset');
  renderCards('cards-cumulative', 'cumulative');

  lastTradesData = d.trades;
  renderTrades();

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

  const slRows = d.stop_loss_counts || [];
  document.querySelector('#stop-loss-counts tbody').innerHTML = slRows.length
    ? slRows.map(s => `<tr><td>${s.coin}</td><td>${s.stop_loss_count}</td><td>${s.liquidation_count}</td><td>${s.total_count}</td></tr>`).join('')
    : '<tr><td colspan="4" style="color:#9aa0a6;">아직 손절/강제청산 기록 없음</td></tr>';
}
load();
setInterval(load, 15000);
setInterval(tickProgressDisplay, 1000);  // "N초 전" 텍스트만 1초마다 갱신 (서버 재요청 없음)
// 분석팀 리포트는 1시간마다만 바뀌므로 5분 간격이면 충분 — 지금 분석팀 탭을
// 보고 있을 때만 갱신한다.
setInterval(() => {
  if (document.getElementById('tab-analysis').classList.contains('active')) loadAnalysis();
}, 5 * 60 * 1000);
// 원장 검증은 거래가 쌓일 때만 결과가 바뀌므로 분석팀 탭과 동일하게 5분 간격,
// 지금 이 탭을 보고 있을 때만 갱신한다.
setInterval(() => {
  if (document.getElementById('tab-verify').classList.contains('active')) loadVerify();
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


@app.route("/api/analysis/history")
def api_analysis_history():
    """과거 전체 회차의 타임스탬프 목록 (최신순)."""
    return jsonify({"timestamps": get_analysis_history_timestamps()})


@app.route("/api/analysis/round/<timestamp>")
def api_analysis_round(timestamp: str):
    """특정 회차의 6개 역할 리포트 본문. timestamp 형식: YYYY-MM-DD_HHMM."""
    if not re.match(r"^\d{4}-\d{2}-\d{2}_\d{4}$", timestamp):
        return jsonify({"error": "잘못된 타임스탬프 형식"}), 400
    return jsonify({"timestamp": timestamp, "reports": get_analysis_round(timestamp)})


@app.route("/api/verify")
def api_verify():
    """거래 원장(trades) ↔ 포트폴리오 문서 정합성 검증 — verify_ledger.py의
    analyze_key()를 그대로 재사용한다(CLI와 대시보드가 서로 다른 로직을 갖지
    않도록 한 곳으로 모음). 읽기 전용 조회이며 trading.db를 수정하지 않는다.

    자동매매 AI(LSTM) 모델 구조 정보도 같이 내려준다 — 실제 학습 이력(시각,
    정확도 추이)은 이미 '모의투자' 탭의 "LSTM 모델 학습 기록" 표가 보여주므로,
    여기서는 코드에 고정된 아키텍처 설정(hidden/layers/피처 등)만 노출한다.
    LSTMClassifier의 실제 기본값을 하드코딩하지 않고 inspect로 직접 읽어서,
    나중에 그 기본값이 바뀌어도 이 라우트를 따로 고칠 필요가 없게 한다."""
    import inspect
    init_params = inspect.signature(lstm_model.LSTMClassifier.__init__).parameters

    with storage.connect() as conn:
        keys = verify_ledger.list_all_keys(conn)
        results = [verify_ledger.analyze_key(conn, k) for k in keys]

    return jsonify({
        "results": results,
        "model_info": {
            "model_type": "lstm",
            "architecture": "단일 레이어 LSTM (PyTorch nn.LSTM 기반 자체 학습 모델 — 외부 사전학습 모델 아님)",
            "hidden_size": init_params["hidden"].default,
            "num_layers": init_params["layers"].default,
            "seq_len_minutes": lstm_model.SEQ_LEN,
            "n_features": lstm_model.N_FEATURES,
            "feature_cols": lstm_model.FEATURE_COLS,
        },
    })


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
            # 조립/역파싱 로직은 storage.parse_portfolio_key()로 통일 (옛 데이터나
            # 예상 밖의 키는 기존과 동일하게 (name, "cumulative")로 폴백한다 —
            # analysis_team 리포트 2026-11-16_1619_storage.md).
            strategy, mode = storage.parse_portfolio_key(name)
            # Portfolio.total_value_krw()를 그대로 재사용한다 — 예전엔 여기서
            # "cash_krw + qty*price"만 직접 더해서 숏 포지션의 마진/미실현손익이
            # 총 평가액에서 통째로 빠져있었다(숏을 들고 있는 전략의 pnl%가 실제보다
            # 낮게 보이는 버그). portfolio.py의 공식 계산식 하나로 통일한다.
            pf = portfolio.Portfolio.from_doc(doc)
            total = pf.total_value_krw(prices)
            initial = doc["initial_capital_krw"]
            pnl = (total / initial - 1.0) * 100

            slots = []
            # doc["slots"]의 원본 dict를 직접 읽지 않고 pf.slots(=CoinSlot 객체)를 쓴다.
            # 숏 관련 필드(short_qty 등)가 나중에 추가된 거라, 그 전에 저장된 실제
            # 운영 DB의 옛 포트폴리오 문서에는 이 키들이 아예 없을 수 있다 — dict를
            # 직접 인덱싱하면 KeyError가 나지만, CoinSlot 데이터클래스는 기본값
            # (0.0)이 있어서 Portfolio.from_doc()을 거치면 안전하게 채워진다.
            for coin, s in pf.slots.items():
                price = prices.get(coin)
                has_long = s.qty > 0
                has_short = s.short_qty > 0
                if has_long:
                    position = "long"
                    unreal_pnl_krw = s.qty * (price - s.avg_cost_krw) if price else None
                    unreal_pnl_pct = ((price / s.avg_cost_krw) - 1.0) * 100 if price and s.avg_cost_krw else None
                elif has_short:
                    position = "short"
                    unreal_pnl_krw = (s.short_avg_entry_krw - price) * s.short_qty if price else None
                    unreal_pnl_pct = ((s.short_avg_entry_krw / price) - 1.0) * 100 if price and s.short_avg_entry_krw else None
                else:
                    position = "none"
                    unreal_pnl_krw = None
                    unreal_pnl_pct = None
                slots.append({
                    "coin": coin,
                    "position": position,
                    "qty": s.qty,
                    "avg_cost_krw": s.avg_cost_krw,
                    "short_qty": s.short_qty,
                    "short_avg_entry_krw": s.short_avg_entry_krw,
                    "short_margin_krw": s.short_margin_krw,
                    "cash_krw": s.cash_krw,
                    "current_price_krw": price,
                    "leverage": portfolio.LEVERAGE if has_short else None,
                    "unrealized_pnl_krw": unreal_pnl_krw,
                    "unrealized_pnl_pct": unreal_pnl_pct,
                })

            portfolios.append({
                "strategy": strategy,
                "mode": mode,
                "total_value_krw": total,
                "initial_capital_krw": initial,
                "pnl_krw": total - initial,
                "pnl_pct": pnl,
                # 수익률만으로는 변동성이 큰 코인에 쏠린 전략이 우연히 좋은
                # 성과로만 보일 수 있어, 고점 대비 낙폭도 같이 노출한다
                # (analysis_team risk 리포트 2026-11-22_0918, synthesizer
                # Top3 #2 — portfolio.py의 peak_value_krw/max_drawdown_pct).
                "max_drawdown_pct": pf.max_drawdown_pct,
                "trade_count": doc.get("trade_count", 0),
                "created_at": doc.get("created_at"),
                "slots": slots,
            })

        trades = storage.recent_trades(conn, limit=200)
        for t in trades:
            # storage.parse_portfolio_key()가 유효하지 않은 키는 (원래 문자열,
            # "cumulative")로 폴백하므로, strategy를 그대로 덮어써도 기존
            # if/else 분기와 동일한 결과가 된다.
            strategy, mode = storage.parse_portfolio_key(t["strategy"])
            t["strategy"] = strategy
            t["mode"] = mode

        # 코인별 최근 10회, 화면엔 시간순(과거->최신)으로 묶어서 내려줘야
        # 프론트에서 "이전 대비 ▲▼" 추이 계산이 바로 된다.
        trainings: list[dict] = []
        for coin in ["BTC", "ETH", "XRP", "SOL", "DOGE", "ADA"]:
            rows = storage.model_training_history(conn, coin=coin, limit=10)
            trainings.extend(reversed(rows))  # 오래된 것 먼저

        prediction_accuracy = storage.prediction_accuracy_by_coin(conn)
        # 코인별 손절/강제청산 발생 횟수 — 변동성 큰 코인이 실제로 더 자주
        # 손절에 걸리는지 데이터로 확인하기 위한 참고용 집계. 트레이딩 로직은
        # 건드리지 않는다 (analysis_team risk 리포트 2026-11-22_0517 구체적
        # 제안 #2, synthesizer Top3 #3).
        stop_loss_counts = storage.stop_loss_counts_by_coin(conn)

    # predictions/model_trainings 보관주기가 실제로 필요한 시점인지 판단할
    # 관측 지표 (analysis_team 리포트 2026-11-16_1619_storage.md 제안).
    # 스키마/데이터는 건드리지 않는 순수 조회이므로 connect() 블록 밖에서도 안전.
    db_size_mb = round(storage.db_file_size_mb(), 2)

    return jsonify({"meta": meta, "progress": progress, "portfolios": portfolios,
                     "trades": trades, "trainings": trainings,
                     "prediction_accuracy": prediction_accuracy,
                     "stop_loss_counts": stop_loss_counts,
                     "db_size_mb": db_size_mb})


if __name__ == "__main__":
    print("대시보드 실행: http://localhost:5000  (같은 와이파이의 폰에서는 이 PC의 사설IP:5000)")
    app.run(host="0.0.0.0", port=5000, debug=False)
