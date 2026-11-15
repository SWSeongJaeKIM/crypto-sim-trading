"""
PC에 상주시켜 계속 돌리는 메인 루프.

Claude/인터넷상의 AI를 전혀 호출하지 않는다. 이 파일 하나를
`python3 main_loop.py` 로 실행해두면(또는 OS 서비스로 등록) 다음을 반복한다:

  - 매 1분: 6코인 최신 1분봉 수집 -> 5개 전략(조건식 4 + LSTM 1)이 각자
    신호 계산 -> 모의 매수/매도 -> SQLite에 상태 저장
  - 매일 1회(기본 새벽 4시): 최근 N일 1분봉으로 LSTM 재학습

중지하려면 Ctrl+C. 로그는 콘솔과 local/run.log에 같이 남는다.
"""

from __future__ import annotations

import logging
import signal
import sys
import time
from datetime import datetime, timezone

import pandas as pd

import data_fetcher
import lstm_model
import storage
from portfolio import Portfolio, COINS, INITIAL_CAPITAL_KRW
from strategies import ALL_STRATEGIES

TRADE_FRACTION = 0.5          # 매수/매도 시 자금의 몇 %를 움직일지
CYCLE_INTERVAL_SEC = 60       # 매매 사이클 주기 (1분봉이므로 60초)
RETRAIN_HOUR_LOCAL = 4        # 매일 이 시(0~23, 로컬 시간)에 LSTM 재학습
HISTORY_DAYS_FOR_TRAIN = 14   # LSTM 학습에 쓸 최근 일수
STRATEGY_NAMES = list(ALL_STRATEGIES.keys()) + ["lstm_predictor"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("run.log", encoding="utf-8")],
)
log = logging.getLogger("main_loop")

_stop = False


def _handle_sigint(signum, frame):
    global _stop
    log.info("종료 신호 수신 — 이번 사이클 끝나면 정지합니다.")
    _stop = True


signal.signal(signal.SIGINT, _handle_sigint)
signal.signal(signal.SIGTERM, _handle_sigint)


def set_progress(conn, stage: str, detail: str = "", **extra) -> None:
    """대시보드가 보여줄 '지금 뭐 하고 있는지' 상태. 매 단계 전환마다 갱신.
    stage: idle | collecting | training | cycle_running | sleeping | error"""
    storage.set_meta(conn, "progress", {
        "stage": stage,
        "detail": detail,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **extra,
    })


def load_or_new_portfolio(conn, strategy_name: str) -> Portfolio:
    doc = storage.get_portfolio(conn, strategy_name)
    if doc:
        return Portfolio.from_doc(doc)
    return Portfolio.new(strategy_name, coins=COINS, initial_capital_krw=INITIAL_CAPITAL_KRW)


def run_one_cycle(conn) -> None:
    now = datetime.now(timezone.utc).isoformat()
    set_progress(conn, "collecting", "최신 1분봉 수집 중...")
    candles = data_fetcher.fetch_all_latest_candles(count=max(lstm_model.SEQ_LEN + 5, 200))
    if not candles:
        log.warning("캔들 수집 실패 — 이번 사이클 스킵")
        set_progress(conn, "error", "캔들 수집 실패, 다음 사이클에 재시도")
        return

    set_progress(conn, "cycle_running", "5개 전략 신호 계산/매매 중...")

    prices_krw = {coin: float(df["close"].iloc[-1]) for coin, df in candles.items() if len(df)}

    # 지난 사이클에 LSTM이 냈던 예측들을 이번 사이클의 실제가로 확정 (적중여부 계산)
    n_resolved = storage.resolve_predictions(conn, prices_krw)
    if n_resolved:
        log.info(f"LSTM 예측 {n_resolved}건 확정")

    for strat_name in STRATEGY_NAMES:
        if strat_name == "lstm_predictor":
            strat = lstm_model.LSTMStrategy(coin=None)  # coin은 코인별로 아래서 바꿔 끼움
        else:
            params = storage.get_params(conn, strat_name) or ALL_STRATEGIES[strat_name].default_params()
            strat = ALL_STRATEGIES[strat_name](**params)

        pf = load_or_new_portfolio(conn, strat_name)

        for coin, df in candles.items():
            if coin not in prices_krw or coin not in pf.slots or df.empty:
                continue
            if strat_name == "lstm_predictor":
                strat.coin = coin
            if len(df) < strat.min_bars():
                continue
            signal_ = strat.signal(df)
            price = prices_krw[coin]

            if strat_name == "lstm_predictor":
                prob_up = lstm_model.predict_proba(coin, df)
                if prob_up is not None:
                    storage.add_prediction(conn, coin, now, prob_up, price)
            if signal_ == "BUY":
                pf.buy(coin, price, pct_of_cash=TRADE_FRACTION, reason=f"{strat_name} BUY")
            elif signal_ == "SELL":
                pf.sell(coin, price, pct_of_qty=TRADE_FRACTION, reason=f"{strat_name} SELL")

        storage.set_portfolio(conn, strat_name, pf.to_doc(), now)
        storage.add_trades(conn, strat_name, pf.new_trades_as_dicts())
        total_value = pf.total_value_krw(prices_krw)
        pnl_pct = pf.pnl_pct(prices_krw)
        storage.add_snapshot(conn, strat_name, now, total_value, pnl_pct)

    storage.set_meta(conn, "status", {"last_run_at": now, "prices_krw": prices_krw, "source": "upbit"})
    log.info(f"사이클 완료 @ {now} | 가격 샘플 BTC={prices_krw.get('BTC')}")


def retrain_lstm(conn) -> None:
    log.info(f"LSTM 재학습 시작 (최근 {HISTORY_DAYS_FOR_TRAIN}일 1분봉)...")
    results = []

    def _on_fetch_progress(coin, i, total):
        set_progress(conn, "training", f"{coin} 데이터 수집 중... ({i}/{total})",
                     current_coin=coin, coin_index=i, coin_total=total, train_step="fetching")

    history = data_fetcher.fetch_all_minute_history(days=HISTORY_DAYS_FOR_TRAIN, on_progress=_on_fetch_progress)

    for i, (coin, df) in enumerate(history.items(), start=1):
        set_progress(conn, "training", f"{coin} 모델 학습 중... ({i}/{len(history)})",
                     current_coin=coin, coin_index=i, coin_total=len(history), train_step="fitting")
        r = lstm_model.train_model(coin, df, epochs=8, history_days=HISTORY_DAYS_FOR_TRAIN)
        results.append(r)
        log.info(f"  {r}")
        # 학습할 때마다 누적 기록 (덮어쓰지 않음) — 대시보드에서 정확도 추이를 보기 위함
        storage.add_model_training(conn, {
            "coin": coin,
            "timestamp": r.get("timestamp", datetime.now(timezone.utc).isoformat()),
            "model_type": "lstm_finetune" if r.get("finetuned") else "lstm_initial",
            "feature_cols": lstm_model.FEATURE_COLS,
            "seq_len": r.get("seq_len", lstm_model.SEQ_LEN),
            "history_days": HISTORY_DAYS_FOR_TRAIN,
            "data_start": r.get("data_start"),
            "data_end": r.get("data_end"),
            "n_samples": r.get("n_samples"),
            "n_train": r.get("n_train"),
            "n_val": r.get("n_val"),
            "val_accuracy": r.get("val_accuracy"),
            "epochs": r.get("epochs"),
            "trained": r.get("trained", False),
            "reason": r.get("reason", ""),
        })

    storage.set_meta(conn, "last_lstm_train", {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
    })
    set_progress(conn, "idle", "학습 완료, 대기 중")


def main():
    log.info("=== crypto-sim-trading 로컬 루프 시작 ===")
    log.info(f"대상 코인: {COINS}")
    log.info(f"전략: {STRATEGY_NAMES}")

    with storage.connect() as conn:
        # 최초 1회: LSTM 모델이 하나도 없으면 바로 학습부터
        has_any_model = any((lstm_model.MODEL_DIR / f"{c}_lstm.pt").exists() for c in COINS)
        if not has_any_model:
            log.info("저장된 LSTM 모델이 없어 최초 학습을 먼저 수행합니다 (몇 분 걸릴 수 있음)...")
            retrain_lstm(conn)

        last_retrain_date = None
        while not _stop:
            try:
                run_one_cycle(conn)
            except Exception as e:
                log.exception(f"사이클 실행 중 에러 (스킵하고 계속): {e}")

            now_local = datetime.now()
            if now_local.hour == RETRAIN_HOUR_LOCAL and last_retrain_date != now_local.date():
                try:
                    retrain_lstm(conn)
                    last_retrain_date = now_local.date()
                except Exception as e:
                    log.exception(f"LSTM 재학습 중 에러: {e}")

            for _ in range(CYCLE_INTERVAL_SEC):
                if _stop:
                    break
                time.sleep(1)

    log.info("=== 루프 정상 종료 ===")


if __name__ == "__main__":
    main()
