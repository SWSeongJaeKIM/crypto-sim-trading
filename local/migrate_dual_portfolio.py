"""
1회성 마이그레이션 스크립트.

main_loop.py가 전략별로 "리셋(7일마다 초기화)" + "누적(계속 유지)" 두 개의
포트폴리오를 동시에 운영하도록 바뀌면서, storage.py의 portfolios 테이블
키가 "ma_cross_rsi" 같은 옛 전략명 대신 "ma_cross_rsi__cumulative" /
"ma_cross_rsi__reset" 형태로 바뀌었다.

이 스크립트는 기존 trading.db에 옛 형식(모드 구분 없는)으로 쌓여있던
데이터를 다음 규칙으로 1회만 이전한다:
  - portfolios, trades, snapshots 테이블의 "ma_cross_rsi" 같은 옛 키를
    전부 "ma_cross_rsi__cumulative" 로 이름 변경 (지금까지 쌓은 역사를
    버리지 않고 "누적" 포트폴리오로 이어간다).
  - "ma_cross_rsi__reset" 은 이 시점에 1000만원으로 새로 시작한다
    (다음 main_loop.py 사이클이 돌면 자동으로 생성됨 — 이 스크립트가
    직접 만들 필요는 없음).

실행법:
  python3 migrate_dual_portfolio.py

이미 "__cumulative"/"__reset" 형식으로 저장된 DB에 대해 실행해도 안전
하다 (이미 새 형식인 행은 건드리지 않음).
"""

from __future__ import annotations

import storage

OLD_STRATEGY_NAMES = ["ma_cross_rsi", "bollinger_rsi", "macd_momentum", "trend_follow", "lstm_predictor"]


def _is_old_format(name: str) -> bool:
    return name in OLD_STRATEGY_NAMES


def migrate() -> None:
    with storage.connect() as conn:
        changed = 0

        rows = conn.execute("SELECT strategy FROM portfolios").fetchall()
        for r in rows:
            old_name = r["strategy"]
            if not _is_old_format(old_name):
                continue
            new_name = f"{old_name}__cumulative"
            exists = conn.execute(
                "SELECT 1 FROM portfolios WHERE strategy = ?", (new_name,)
            ).fetchone()
            if exists:
                print(f"  스킵: {new_name} 이 이미 존재 (덮어쓰지 않음)")
                continue
            conn.execute("UPDATE portfolios SET strategy = ? WHERE strategy = ?", (new_name, old_name))
            conn.execute("UPDATE trades SET strategy = ? WHERE strategy = ?", (new_name, old_name))
            conn.execute("UPDATE snapshots SET strategy = ? WHERE strategy = ?", (new_name, old_name))
            print(f"  이전: {old_name} -> {new_name}")
            changed += 1

        if changed == 0:
            print("이전할 옛 형식 데이터가 없습니다 (이미 마이그레이션 됐거나 처음 실행하는 환경).")
        else:
            print(f"\n완료: {changed}개 전략의 기존 데이터를 '__cumulative'로 이전했습니다.")
            print("'__reset' 포트폴리오들은 main_loop.py를 다시 실행하면 1000만원으로 새로 시작됩니다.")


if __name__ == "__main__":
    migrate()
