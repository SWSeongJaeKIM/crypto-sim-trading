"""
로컬 SQLite 기반 상태 저장소.

Claude Artifact DB(클라우드) 대신, 사용자 PC에 상주하는 SQLite 파일 하나로
포트폴리오/거래내역/스냅샷/모델 메타정보를 저장한다. 외부 네트워크,
Claude API 전부 쓰지 않음 — 순수 로컬 파일 I/O.

테이블:
  portfolios(strategy TEXT PRIMARY KEY, data_json TEXT, updated_at TEXT)
  trades(id INTEGER PK AUTOINCREMENT, strategy TEXT, coin TEXT, side TEXT,
         qty REAL, price_krw REAL, reason TEXT, timestamp TEXT)
  snapshots(id INTEGER PK AUTOINCREMENT, strategy TEXT, timestamp TEXT,
            total_value_krw REAL, pnl_pct REAL)
  meta(key TEXT PRIMARY KEY, value_json TEXT)
  model_trainings(id INTEGER PK AUTOINCREMENT, coin TEXT, timestamp TEXT,
                   model_type TEXT, feature_cols_json TEXT, seq_len INTEGER,
                   history_days INTEGER, data_start TEXT, data_end TEXT,
                   n_samples INTEGER, n_train INTEGER, n_val INTEGER,
                   val_accuracy REAL, epochs INTEGER, trained INTEGER,
                   reason TEXT)
    — LSTM을 재학습할 때마다 1행씩 누적 (덮어쓰지 않음). "이 모델이 언제,
    어떤 기간의 데이터로, 몇 개 샘플로 학습됐고 정확도가 얼마였는지"를
    시간순으로 전부 보존해서, 대시보드에서 모델 성능 추이를 볼 수 있게 한다.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).parent / "trading.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS portfolios (
    strategy TEXT PRIMARY KEY,
    data_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy TEXT NOT NULL,
    coin TEXT NOT NULL,
    side TEXT NOT NULL,
    qty REAL NOT NULL,
    price_krw REAL NOT NULL,
    reason TEXT,
    timestamp TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy, timestamp DESC);
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    total_value_krw REAL NOT NULL,
    pnl_pct REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_snapshots_strategy ON snapshots(strategy, timestamp DESC);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model_trainings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    coin TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    model_type TEXT NOT NULL,
    feature_cols_json TEXT NOT NULL,
    seq_len INTEGER NOT NULL,
    history_days INTEGER,
    data_start TEXT,
    data_end TEXT,
    n_samples INTEGER,
    n_train INTEGER,
    n_val INTEGER,
    val_accuracy REAL,
    epochs INTEGER,
    trained INTEGER NOT NULL,
    reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_model_trainings_coin ON model_trainings(coin, timestamp DESC);
"""


@contextmanager
def connect(db_path: Path | str = DB_PATH):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def get_portfolio(conn, strategy: str) -> dict | None:
    row = conn.execute("SELECT data_json FROM portfolios WHERE strategy = ?", (strategy,)).fetchone()
    return json.loads(row["data_json"]) if row else None


def get_all_portfolios(conn) -> dict[str, dict]:
    rows = conn.execute("SELECT strategy, data_json FROM portfolios").fetchall()
    return {r["strategy"]: json.loads(r["data_json"]) for r in rows}


def set_portfolio(conn, strategy: str, doc: dict, timestamp: str) -> None:
    conn.execute(
        """INSERT INTO portfolios (strategy, data_json, updated_at) VALUES (?, ?, ?)
           ON CONFLICT(strategy) DO UPDATE SET data_json=excluded.data_json, updated_at=excluded.updated_at""",
        (strategy, json.dumps(doc), timestamp),
    )


def add_trades(conn, strategy: str, trades: list[dict]) -> None:
    for t in trades:
        conn.execute(
            "INSERT INTO trades (strategy, coin, side, qty, price_krw, reason, timestamp) VALUES (?,?,?,?,?,?,?)",
            (strategy, t["coin"], t["side"], t["qty"], t["price_krw"], t.get("reason", ""), t["timestamp"]),
        )


def add_snapshot(conn, strategy: str, timestamp: str, total_value_krw: float, pnl_pct: float) -> None:
    conn.execute(
        "INSERT INTO snapshots (strategy, timestamp, total_value_krw, pnl_pct) VALUES (?,?,?,?)",
        (strategy, timestamp, total_value_krw, pnl_pct),
    )


def recent_snapshots(conn, strategy: str, limit: int = 30) -> list[dict]:
    """시간순(오래된 것 먼저)으로 반환 — learner.recent_performance가 기대하는 순서."""
    rows = conn.execute(
        "SELECT timestamp, total_value_krw, pnl_pct FROM snapshots "
        "WHERE strategy = ? ORDER BY timestamp DESC LIMIT ?",
        (strategy, limit),
    ).fetchall()
    rows = list(reversed(rows))
    return [{"timestamp": r["timestamp"], "total_value_krw": r["total_value_krw"], "pnl_pct": r["pnl_pct"]} for r in rows]


def recent_trades(conn, strategy: str | None = None, limit: int = 20) -> list[dict]:
    if strategy:
        rows = conn.execute(
            "SELECT * FROM trades WHERE strategy = ? ORDER BY timestamp DESC LIMIT ?",
            (strategy, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_meta(conn, key: str) -> dict | None:
    row = conn.execute("SELECT value_json FROM meta WHERE key = ?", (key,)).fetchone()
    return json.loads(row["value_json"]) if row else None


def set_meta(conn, key: str, value: dict) -> None:
    conn.execute(
        """INSERT INTO meta (key, value_json) VALUES (?, ?)
           ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json""",
        (key, json.dumps(value)),
    )


def get_params(conn, strategy: str) -> dict | None:
    return get_meta(conn, f"params:{strategy}")


def set_params(conn, strategy: str, params: dict) -> None:
    set_meta(conn, f"params:{strategy}", params)


def add_model_training(conn, record: dict) -> None:
    """LSTM 학습(최초 학습 또는 파인튜닝) 1회를 model_trainings 테이블에 누적 기록.
    기존 기록을 덮어쓰지 않음 — 학습할 때마다 새 행이 쌓여서 추이를 볼 수 있다.
    record 예시 키: coin, timestamp, model_type, feature_cols, seq_len, history_days,
                    data_start, data_end, n_samples, n_train, n_val, val_accuracy,
                    epochs, trained, reason"""
    conn.execute(
        """INSERT INTO model_trainings
           (coin, timestamp, model_type, feature_cols_json, seq_len, history_days,
            data_start, data_end, n_samples, n_train, n_val, val_accuracy, epochs,
            trained, reason)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            record["coin"], record["timestamp"], record.get("model_type", "lstm"),
            json.dumps(record.get("feature_cols", [])), record.get("seq_len"),
            record.get("history_days"), record.get("data_start"), record.get("data_end"),
            record.get("n_samples"), record.get("n_train"), record.get("n_val"),
            record.get("val_accuracy"), record.get("epochs"),
            1 if record.get("trained") else 0, record.get("reason", ""),
        ),
    )


def model_training_history(conn, coin: str | None = None, limit: int = 50) -> list[dict]:
    """학습 기록 조회, 최신순. coin 생략하면 전체 코인."""
    if coin:
        rows = conn.execute(
            "SELECT * FROM model_trainings WHERE coin = ? ORDER BY timestamp DESC LIMIT ?",
            (coin, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM model_trainings ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["feature_cols"] = json.loads(d.pop("feature_cols_json"))
        out.append(d)
    return out
