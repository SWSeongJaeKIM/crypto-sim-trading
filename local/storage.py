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
  predictions(id INTEGER PK AUTOINCREMENT, coin TEXT, predicted_at TEXT,
              prob_up REAL, predicted_direction TEXT, price_at_prediction REAL,
              resolved INTEGER, actual_price REAL, correct INTEGER)
    — LSTM이 실시간으로 매 사이클 "다음 1분 상승확률"을 추론할 때마다 1행
    기록 (학습 때의 val_accuracy는 과거 데이터 기준이고, 이건 "실전에서
    실제로 맞았는지"를 보여준다). 다음 사이클에서 그 시점 실제가를 채워
    넣어 적중여부(correct)를 계산 — 이게 '예측 적중률'의 재료가 된다.
"""

from __future__ import annotations

import json
import time
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
-- recent_trades(strategy=None)이 호출하는 "strategy 필터 없는 전체 조회"
-- (대시보드의 "전체 최근 거래", dashboard.py가 매 새로고침마다 호출)는
-- idx_trades_strategy의 선두 컬럼(strategy)을 못 써서 전체 스캔이 된다.
-- timestamp 단독 인덱스를 추가해 그 경로가 인덱스로 처리되게 한다.
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp DESC);
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
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    coin TEXT NOT NULL,
    predicted_at TEXT NOT NULL,
    prob_up REAL NOT NULL,
    predicted_direction TEXT NOT NULL,
    price_at_prediction REAL NOT NULL,
    resolved INTEGER NOT NULL DEFAULT 0,
    actual_price REAL,
    correct INTEGER
);
CREATE INDEX IF NOT EXISTS idx_predictions_coin ON predictions(coin, predicted_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_unresolved ON predictions(resolved);
"""


_BUSY_TIMEOUT_MS = 5000
_MAX_RETRIES = 8
_RETRY_BACKOFF_SEC = 0.3  # 재시도마다 선형으로 늘어남: 0.3, 0.6, 0.9, ...

# executescript(SCHEMA)의 CREATE TABLE/INDEX IF NOT EXISTS들은 테이블이 이미
# 있어도 매번 sqlite_master를 확인하면서 자체 트랜잭션을 여는데, 이게 매
# connect()마다(사이클마다, progress 기록마다) 반복되면 그 자체가 "database
# is locked" 경합 기회를 늘린다. 같은 DB 파일 기준으로 스키마는 프로세스당
# 딱 한 번만 적용하면 되므로, 이미 적용했던 경로는 다시 실행하지 않는다.
_schema_initialized_paths: set[str] = set()


def _retry_on_locked(fn, *args, **kwargs):
    """fn(*args, **kwargs)을 실행하다가 'database is locked'를 만나면 짧게
    쉬고 재시도. SQLite의 PRAGMA busy_timeout이 1차 방어선이고, 이건 그
    이후에도 못 뚫은 경우를 위한 2차 방어선이다 (Windows에서 짧은 생명주기의
    커넥션들이 아주 촘촘하게 몰릴 때 busy_timeout 안에서도 즉시 에러가
    올라오는 경우가 실제로 관찰됐다)."""
    attempt = 0
    while True:
        try:
            return fn(*args, **kwargs)
        except sqlite3.OperationalError as e:
            if "locked" not in str(e).lower() or attempt >= _MAX_RETRIES:
                raise
            attempt += 1
            time.sleep(_RETRY_BACKOFF_SEC * attempt)


class _RetryingConnection(sqlite3.Connection):
    """execute/executemany/executescript/commit 호출마다 'database is
    locked'를 자동으로 재시도하는 커넥션. storage.py의 모든 함수는
    'conn.execute(...)' 형태로 이 객체를 그대로 쓰므로, 호출부를 하나도
    바꾸지 않고 재시도를 전역으로 적용할 수 있다."""

    def execute(self, *args, **kwargs):
        return _retry_on_locked(super().execute, *args, **kwargs)

    def executemany(self, *args, **kwargs):
        return _retry_on_locked(super().executemany, *args, **kwargs)

    def executescript(self, *args, **kwargs):
        return _retry_on_locked(super().executescript, *args, **kwargs)

    def commit(self, *args, **kwargs):
        return _retry_on_locked(super().commit, *args, **kwargs)


@contextmanager
def connect(db_path: Path | str = DB_PATH):
    # main_loop(사이클/재학습 중 진행상태 기록)와 dashboard(주기적으로 읽기),
    # 그리고 retrain_now.py를 수동으로 같이 돌리는 경우까지 — 여러 프로세스가
    # 동시에 같은 DB 파일을 열 수 있다. 기본 journal_mode로는 "database is
    # locked" 에러가 바로 날 수 있다. WAL 모드는 읽기/쓰기가 서로 거의
    # 블로킹하지 않게 해주고, busy_timeout은 그래도 충돌하면 에러 던지기
    # 전에 잠깐 재시도 대기한다. _RetryingConnection이 그 위에 애플리케이션
    # 레벨 재시도까지 한 번 더 걸어준다.
    key = str(Path(db_path).resolve())

    conn = sqlite3.connect(
        str(db_path), timeout=_BUSY_TIMEOUT_MS / 1000, factory=_RetryingConnection
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
    try:
        if key not in _schema_initialized_paths:
            conn.executescript(SCHEMA)
            conn.commit()
            _schema_initialized_paths.add(key)

        yield conn
        conn.commit()
    except Exception:
        # with 블록 안에서 예외가 나면 커밋 전에 여기로 온다. 커밋 안 된
        # 트랜잭션을 들고 그냥 close()하면 자동 롤백되긴 하지만, 그 사이에
        # 다른 프로세스가 이 커넥션의 쓰기 잠금을 기다리다 "database is
        # locked"를 만날 수 있으므로 즉시 명시적으로 롤백해 잠금을 빨리 푼다.
        conn.rollback()
        raise
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


def add_prediction(conn, coin: str, predicted_at: str, prob_up: float, price_at_prediction: float) -> None:
    """LSTM이 이번 사이클에 낸 '다음 1분 상승확률' 예측 1건을 기록 (미확정 상태).
    다음 사이클에서 resolve_predictions()가 실제가를 채워 적중여부를 계산한다."""
    direction = "UP" if prob_up >= 0.5 else "DOWN"
    conn.execute(
        """INSERT INTO predictions (coin, predicted_at, prob_up, predicted_direction, price_at_prediction, resolved)
           VALUES (?,?,?,?,?,0)""",
        (coin, predicted_at, prob_up, direction, price_at_prediction),
    )


def resolve_predictions(conn, current_prices_krw: dict[str, float]) -> int:
    """아직 확정 안 된(resolved=0) 예측들 중, 예측 시점과 다른(=1사이클 지난) 것들을
    현재가로 확정짓는다. 예측방향(UP/DOWN)과 실제 가격변화 방향이 일치하면 correct=1.
    반환: 이번에 확정한 건수."""
    rows = conn.execute("SELECT * FROM predictions WHERE resolved = 0").fetchall()
    # 미확정 건마다 개별 UPDATE 왕복을 내지 않고, 한 번에 적용할 변경을 모아
    # executemany로 한 번에 묶어서 실행한다 (가격 피드 지연 등으로 미확정 건이
    # 일시적으로 쌓였을 때 N번 왕복이 생기는 걸 막는다). 결과는 기존과 동일.
    updates = []
    for r in rows:
        coin = r["coin"]
        if coin not in current_prices_krw:
            continue
        actual_price = current_prices_krw[coin]
        if actual_price == r["price_at_prediction"]:
            continue  # 가격이 안 바뀐 사이클이면 다음 기회에 확정 (방향 판단 불가)
        actual_direction = "UP" if actual_price > r["price_at_prediction"] else "DOWN"
        correct = 1 if actual_direction == r["predicted_direction"] else 0
        updates.append((actual_price, correct, r["id"]))
    if updates:
        conn.executemany(
            "UPDATE predictions SET resolved=1, actual_price=?, correct=? WHERE id=?",
            updates,
        )
    return len(updates)


def prediction_accuracy(conn, coin: str | None = None, limit: int = 500) -> dict:
    """확정된(resolved=1) 예측들의 적중률 요약. coin 생략하면 전체 코인 합산."""
    if coin:
        rows = conn.execute(
            "SELECT correct FROM predictions WHERE resolved=1 AND coin=? ORDER BY predicted_at DESC LIMIT ?",
            (coin, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT correct FROM predictions WHERE resolved=1 ORDER BY predicted_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    total = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    return {"coin": coin, "total": total, "correct": correct,
            "accuracy": round(correct / total, 4) if total else None}


def prediction_accuracy_by_coin(conn, limit_per_coin: int = 500) -> list[dict]:
    coins = [r["coin"] for r in conn.execute("SELECT DISTINCT coin FROM predictions").fetchall()]
    return [prediction_accuracy(conn, coin=c, limit=limit_per_coin) for c in coins]
