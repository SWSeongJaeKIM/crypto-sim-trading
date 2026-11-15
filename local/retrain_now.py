"""
지금 바로 LSTM 재학습(또는 최초학습)을 1회 수동 실행하는 스크립트.

main_loop.py는 보통 새벽 4시에만 자동으로 재학습하는데, 그 전에
"학습이 실제로 잘 되는지", "학습기록이 대시보드에 잘 쌓이는지"를
바로 확인하고 싶을 때 이걸 실행하면 된다.

main_loop.py가 이미 돌고 있어도 상관없이 따로 실행 가능 (같은 trading.db를
공유, 서로 충돌 없음 — 각자 자기 차례에 읽고 쓰기만 함).

실행:
    python3 retrain_now.py
"""

from __future__ import annotations

import storage
from main_loop import retrain_lstm

if __name__ == "__main__":
    with storage.connect() as conn:
        retrain_lstm(conn)
    print("재학습 완료. 대시보드를 새로고침해서 '학습 기록' 표를 확인하세요.")
