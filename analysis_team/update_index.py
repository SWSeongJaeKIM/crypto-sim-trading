"""
reports/ 폴더에 쌓인 역할별 리포트들 중, 역할별 '가장 최근 것'만 모아서
reports/latest.json을 갱신한다.

스케줄 작업이 6개 역할의 리포트(.md)를 reports/에 다 쓴 뒤, 이 스크립트를
한 번 실행하면 된다. 대시보드(local/dashboard.py)는 GitHub에서
reports/latest.json 하나만 받아오면 6개 역할의 최신 리포트 경로와 한 줄
요약을 바로 알 수 있다 — 매번 reports/ 디렉토리 전체를 뒤질 필요 없음.

파일명 규칙: reports/<YYYY-MM-DD_HHmm>_<role>.md
role 종류: strategy, lstm, execution, storage, risk, synthesizer

사용법:
  python3 update_index.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPORTS_DIR = Path(__file__).parent / "reports"
INDEX_PATH = REPORTS_DIR / "latest.json"

ROLE_LABELS = {
    "strategy": "전략 분석가",
    "lstm": "LSTM 분석가",
    "execution": "실행/버그 분석가",
    "storage": "데이터/저장소 분석가",
    "risk": "리스크 분석가",
    "synthesizer": "종합 조정자",
}

FILENAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}_\d{4})_([a-z_]+)\.md$")


def extract_summary(text: str) -> str:
    """리포트 본문에서 '## 한줄 요약' 또는 '## 이번 회차 Top 3 개선 제안' 다음 줄을 뽑아 요약으로 쓴다."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("## 한줄 요약") or line.strip().startswith("## 이번 회차"):
            for nxt in lines[i + 1:]:
                nxt = nxt.strip().lstrip("-").strip()
                if nxt:
                    return nxt[:160]
    # 못 찾으면 첫 비어있지 않은 줄
    for line in lines:
        if line.strip() and not line.strip().startswith("#"):
            return line.strip()[:160]
    return "(요약 없음)"


def build_index() -> dict:
    latest_by_role: dict[str, dict] = {}
    if not REPORTS_DIR.exists():
        return {"generated_at": None, "roles": {}}

    for path in sorted(REPORTS_DIR.glob("*.md")):
        m = FILENAME_RE.match(path.name)
        if not m:
            continue
        timestamp, role = m.group(1), m.group(2)
        # sorted()로 이미 timestamp 오름차순이므로, 마지막에 덮어쓴 게 최신
        text = path.read_text(encoding="utf-8")
        latest_by_role[role] = {
            "role": role,
            "role_label": ROLE_LABELS.get(role, role),
            "timestamp": timestamp,
            "path": f"analysis_team/reports/{path.name}",
            "summary": extract_summary(text),
        }

    newest_ts = max((r["timestamp"] for r in latest_by_role.values()), default=None)
    return {"generated_at": newest_ts, "roles": latest_by_role}


def main() -> None:
    index = build_index()
    INDEX_PATH.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"갱신됨: {INDEX_PATH} (역할 {len(index['roles'])}개)")


if __name__ == "__main__":
    main()
