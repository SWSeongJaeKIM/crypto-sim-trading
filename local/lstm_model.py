"""
1분봉 기준 "다음 캔들 상승/하락" 을 예측하는 LSTM 분류 모델.

조건식 4개 전략과 같은 리더보드에서 경쟁시키기 위해, 이 모듈도
strategies.py의 전략 클래스와 동일한 인터페이스(signal(df) -> "BUY"/"SELL"/"HOLD")를
제공하는 LSTMStrategy 클래스를 노출한다.

설계:
- 입력 피처(시점별): close 변화율, volume 변화율, RSI(14), MACD histogram, 볼린저밴드 %b
  (전부 strategies.py의 지표 헬퍼로 계산 — 조건식과 동일한 지표를 공유해서 일관성 유지)
- 시퀀스 길이: 과거 60분 (SEQ_LEN)
- 라벨: 다음 1분봉 종가가 현재 종가보다 오르면 1(UP), 내리면 0(DOWN)
  (동률/거의 동률 구간은 완충을 위해 NEUTRAL_BAND 이내면 학습 라벨에서 제외)
- 출력: 다음 캔들이 오를 확률(0~1). PROB_BUY_MIN 이상이면 BUY,
  PROB_SELL_MAX 이하면 SELL, 그 사이는 HOLD (조건식처럼 신호를 내도록 임계값으로 변환)

학습(train_model)과 추론(predict_proba / signal)을 분리:
- 학습은 하루 1회(메인 루프가 호출)만 돌리고 디스크에 모델 파일로 저장
- 추론(1분마다)은 저장된 가중치를 로드해서 forward pass만 수행 (가벼움, CPU로 충분)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from strategies import rsi, macd, bollinger

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

SEQ_LEN = 60          # 과거 60분을 보고 다음 1분을 예측
NEUTRAL_BAND = 0.0005  # 0.05% 이내 변동은 학습 라벨에서 제외 (노이즈 방지)
PROB_BUY_MIN = 0.55    # 상승확률 55% 이상이면 BUY
PROB_SELL_MAX = 0.45   # 상승확률 45% 이하면 SELL (=하락확률 55% 이상)

FEATURE_COLS = ["ret", "vol_chg", "rsi14", "macd_hist", "bb_pctb"]
N_FEATURES = len(FEATURE_COLS)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV df -> 모델 입력 피처 df. df는 open_time 기준 시간순 정렬 가정."""
    close = df["close"]
    volume = df["volume"]

    ret = close.pct_change()
    vol_chg = volume.pct_change().replace([np.inf, -np.inf], 0.0)
    r = rsi(close, 14) / 100.0  # 0~1로 정규화
    _, _, hist = macd(close, fast=12, slow=26, signal=9)
    macd_hist = hist / close  # 가격 스케일 영향 제거
    upper, mid, lower = bollinger(close, period=20, num_std=2)
    bb_pctb = (close - lower) / (upper - lower).replace(0, np.nan)

    feat = pd.DataFrame({
        "ret": ret,
        "vol_chg": vol_chg,
        "rsi14": r,
        "macd_hist": macd_hist,
        "bb_pctb": bb_pctb,
    })
    return feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_dataset(df: pd.DataFrame, seq_len: int = SEQ_LEN) -> tuple[np.ndarray, np.ndarray]:
    """시퀀스(X) / 라벨(y) 생성. y: 1=다음캔들상승, 0=하락 (중립구간은 제외).

    시퀀스는 시점 i를 "마지막으로 아는 시점"으로 포함해서(arr[i-seq_len+1:i+1])
    i->i+1로의 변화를 예측하도록 만든다. predict_proba()의 추론 입력
    (feat[-seq_len:], 즉 마지막 알려진 시점까지 포함한 윈도우로 그 다음을
    예측)과 정의를 맞추기 위함 — 이전에는 시퀀스가 i-1까지만 포함해서
    (arr[i-seq_len:i]) 학습 시 시퀀스 끝과 라벨 사이에 1개 캔들 간극이
    있었고, 그 결과 모델이 실제 운용 때와 다른 과제로 학습되고 있었다."""
    feat = build_features(df)
    close = df["close"].to_numpy()
    arr = feat.to_numpy(dtype=np.float32)

    xs, ys = [], []
    n = len(df)
    for i in range(seq_len - 1, n - 1):
        next_ret = (close[i + 1] - close[i]) / close[i]
        if abs(next_ret) < NEUTRAL_BAND:
            continue
        xs.append(arr[i - seq_len + 1: i + 1])
        ys.append(1 if next_ret > 0 else 0)

    if not xs:
        return np.empty((0, seq_len, N_FEATURES), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.stack(xs), np.array(ys, dtype=np.float32)


class LSTMClassifier(nn.Module):
    def __init__(self, n_features: int = N_FEATURES, hidden: int = 32, layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)  # logit


HISTORY_DIR = MODEL_DIR / "history"
HISTORY_DIR.mkdir(exist_ok=True, parents=True)


def train_model(coin: str, df: pd.DataFrame, epochs: int = 8, batch_size: int = 256,
                 lr: float = 1e-3, seq_len: int = SEQ_LEN, val_frac: float = 0.15,
                 finetune: bool = True, history_days: int | None = None) -> dict:
    """한 코인에 대해 LSTM을 학습(또는 파인튜닝)하고 모델 파일로 저장.

    finetune=True(기본): 기존 저장된 가중치가 있으면 그걸 불러와 이어서 학습
    (추가학습). 없으면 새로 초기화해서 학습(최초 학습과 동일).
    finetune=False: 항상 처음부터 새로 학습.

    저장은 두 곳에: 최신 가중치는 {coin}_lstm.pt(추론용, 매번 덮어씀),
    그리고 history/{coin}_lstm_YYYY-MM-DD.pt 로 그날 날짜 사본도 같이 남긴다
    (하루 여러 번 재학습해도 그날 파일은 마지막 결과로 덮어씀 — 날짜당 1개 유지)."""
    X, y = build_dataset(df, seq_len=seq_len)
    timestamp = pd.Timestamp.now(tz="UTC").isoformat()
    data_start = str(df["open_time"].iloc[0]) if len(df) else None
    data_end = str(df["open_time"].iloc[-1]) if len(df) else None

    if len(X) < 500:
        return {
            "coin": coin, "trained": False,
            "reason": f"데이터 부족 (샘플 {len(X)}개, 최소 500 필요)",
            "timestamp": timestamp, "data_start": data_start, "data_end": data_end,
            "history_days": history_days, "seq_len": seq_len,
        }

    n = len(X)
    split = int(n * (1 - val_frac))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = LSTMClassifier()
    existing_path = MODEL_DIR / f"{coin}_lstm.pt"
    resumed_from_existing = finetune and existing_path.exists()
    if resumed_from_existing:
        model.load_state_dict(torch.load(existing_path, map_location="cpu"))

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val)

    model.train()
    n_batches = max(1, len(X_train) // batch_size)
    for epoch in range(epochs):
        perm = torch.randperm(len(X_train))
        for b in range(n_batches):
            idx = perm[b * batch_size: (b + 1) * batch_size]
            if len(idx) == 0:
                continue
            opt.zero_grad()
            logits = model(X_train_t[idx])
            loss = loss_fn(logits, y_train_t[idx])
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t)
        val_pred = (torch.sigmoid(val_logits) > 0.5).float()
        val_acc = (val_pred == y_val_t).float().mean().item()

    torch.save(model.state_dict(), existing_path)
    date_str = pd.Timestamp.now().strftime("%Y-%m-%d")
    torch.save(model.state_dict(), HISTORY_DIR / f"{coin}_lstm_{date_str}.pt")

    _loaded_models.pop(coin, None)  # 추론 캐시 무효화 (다음 predict_proba가 새 가중치 로드)

    return {
        "coin": coin,
        "trained": True,
        "finetuned": resumed_from_existing,
        "n_samples": n,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "val_accuracy": round(val_acc, 4),
        "epochs": epochs,
        "timestamp": timestamp,
        "data_start": data_start,
        "data_end": data_end,
        "history_days": history_days,
        "seq_len": seq_len,
    }


def train_all(candles: dict[str, pd.DataFrame], **kwargs) -> list[dict]:
    return [train_model(coin, df, **kwargs) for coin, df in candles.items()]


_loaded_models: dict[str, LSTMClassifier] = {}


def _load_model(coin: str) -> LSTMClassifier | None:
    path = MODEL_DIR / f"{coin}_lstm.pt"
    if not path.exists():
        return None
    if coin not in _loaded_models:
        model = LSTMClassifier()
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        _loaded_models[coin] = model
    return _loaded_models[coin]


def predict_proba(coin: str, df: pd.DataFrame, seq_len: int = SEQ_LEN) -> float | None:
    """최근 seq_len분 데이터로 다음 캔들 상승확률을 추론. 모델 없거나 데이터 부족하면 None."""
    model = _load_model(coin)
    if model is None or len(df) < seq_len:
        return None
    feat = build_features(df)
    window = feat.to_numpy(dtype=np.float32)[-seq_len:].copy()
    x = torch.from_numpy(window).unsqueeze(0)  # (1, seq_len, n_features)
    with torch.no_grad():
        logit = model(x)
        prob_up = torch.sigmoid(logit).item()
    return prob_up


class LSTMStrategy:
    """strategies.py의 Strategy 클래스와 동일한 인터페이스 (signal(df) -> BUY/SELL/HOLD).
    조건식 전략들과 같은 리더보드에서 경쟁시키기 위한 래퍼."""
    name = "lstm_predictor"

    def __init__(self, coin: str, **params):
        self.coin = coin
        self.params = params

    def min_bars(self) -> int:
        return SEQ_LEN + 1

    def signal(self, df: pd.DataFrame) -> str:
        if len(df) < self.min_bars():
            return "HOLD"
        prob_up = predict_proba(self.coin, df)
        if prob_up is None:
            return "HOLD"  # 아직 학습된 모델 없음
        if prob_up >= PROB_BUY_MIN:
            return "BUY"
        if prob_up <= PROB_SELL_MAX:
            return "SELL"
        return "HOLD"

    def signal_with_confidence(self, df: pd.DataFrame) -> tuple[str, float]:
        """prob_up이 0.5(완전 모호)에서 얼마나 멀리 떨어져 있는지를 확신도로
        쓴다. 임계값(0.55/0.45)을 막 넘은 경우는 확신도가 낮고, 0.9나 0.1처럼
        극단으로 갈수록 1.0에 가까워진다. main_loop.py가 이 값으로 베팅
        크기를 정한다."""
        if len(df) < self.min_bars():
            return "HOLD", 0.0
        prob_up = predict_proba(self.coin, df)
        if prob_up is None:
            return "HOLD", 0.0
        if prob_up >= PROB_BUY_MIN:
            sig = "BUY"
        elif prob_up <= PROB_SELL_MAX:
            sig = "SELL"
        else:
            return "HOLD", 0.0
        # 임계값(0.05만큼 0.5에서 떨어진 지점)부터 극단(0.5만큼 떨어진 지점,
        # 즉 prob_up=1.0 또는 0.0)까지를 0~1로 선형 스케일.
        distance_from_mid = abs(prob_up - 0.5)
        threshold_distance = 0.5 - PROB_BUY_MIN  # = 0.05
        confidence = (distance_from_mid - threshold_distance) / (0.5 - threshold_distance)
        return sig, float(np.clip(confidence, 0.0, 1.0))

    @staticmethod
    def default_params() -> dict:
        return {}


if __name__ == "__main__":
    # 합성 데이터로 학습/추론 파이프라인 자체 동작 확인 (네트워크 없이)
    np.random.seed(0)
    n = 3000
    t = pd.date_range("2026-01-01", periods=n, freq="min")
    price = 100 + np.cumsum(np.random.randn(n) * 0.1)
    price = np.clip(price, 1, None)
    df = pd.DataFrame({
        "open_time": t, "open": price, "high": price * 1.0005,
        "low": price * 0.9995, "close": price, "volume": np.random.rand(n) * 10 + 1,
    })

    result = train_model("TEST", df, epochs=3)
    print("학습 결과:", result)

    strat = LSTMStrategy(coin="TEST")
    sig = strat.signal(df)
    print("추론 신호:", sig)
