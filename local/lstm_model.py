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
- 학습은 메인 루프(main_loop.py의 RETRAIN_TIMES_LOCAL)가 하루 10회(약 2시간24분
  간격) 호출하며, 매 호출마다 최근 HISTORY_DAYS_FOR_TRAIN(90)일치 1분봉으로
  파인튠한 뒤 디스크에 모델 파일로 저장
- 추론(1분마다)은 저장된 가중치를 로드해서 forward pass만 수행 (가벼움, CPU로 충분)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from strategies import rsi, macd, bollinger

log = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

# GPU(CUDA)가 있으면 학습에 사용 — 90일치 데이터(코인당 약 13만 샘플) x 6코인
# x 8epoch이라 느린 PC에서는 체감 차이가 크다. 없으면 자동으로 CPU로
# 폴백하므로 이 모듈은 GPU 유무와 상관없이 그대로 동작한다. 추론
# (predict_proba, 매분 호출)은 forward pass 한 번뿐이라 원래부터 CPU로도
# 충분하지만, 모델이 이미 GPU에 있으니 추론도 같은 디바이스에서 수행한다.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQ_LEN = 60          # 과거 60분을 보고 다음 1분을 예측
NEUTRAL_BAND = 0.0005  # 0.05% 이내 변동은 학습 라벨에서 제외 (노이즈 방지)

# 실전 재학습에서 관찰된 val_accuracy가 대개 54~62% 수준(동전 던지기보다
# 아주 조금 나은 정도)이라, 0.55/0.45처럼 느슨한 임계값에서는 모델이
# "약간 오를 것 같다"고 한 거의 랜덤에 가까운 확률 차이로도 매번 BUY/SELL을
# 쐈다. 그 결과 거래 건수가 다른 전략들(6~51건)보다 훨씬 많은 166건까지
# 늘어났고, 3배 레버리지 하에서 "맞춰도 조금 벌고 틀리면 크게 잃는" 구조가
# 반복돼 방향 적중률은 그럭저럭인데 실제 손익은 크게 마이너스가 나는
# 괴리가 생겼다. 임계값을 0.60/0.40으로 높여서, 모델이 꽤 확신하는
# (그래서 노이즈일 가능성이 더 낮은) 신호에서만 매매하도록 좁혔다.
PROB_BUY_MIN = 0.60    # 상승확률 60% 이상이면 BUY (기존 0.55 -> 과매매/노이즈 매매 축소)
PROB_SELL_MAX = 0.40   # 상승확률 40% 이하면 SELL (기존 0.45)

# 파인튠(이어학습) 후 val_accuracy가 이전 저장 모델보다 이 값(절대 비율) 이상
# 떨어지면 저장을 건너뛰고 기존 가중치를 그대로 유지한다. 하루 10회 무인
# 자동 재학습 루프에서 노이즈가 큰 구간을 학습해 성능이 떨어진 가중치가
# 나와도 그대로 다음 추론(실거래 신호)에 쓰이는 걸 막기 위한 안전망.
# 너무 낮게 잡으면 그냥 어려운 시장 상황이었던 정상적 재학습까지 막아
# 가중치가 과거에 고착(stale)될 수 있어 보수적으로 0.03으로 시작.
VAL_ACC_ROLLBACK_THRESHOLD = 0.03

# 위 VAL_ACC_ROLLBACK_THRESHOLD 비교는 원래 val_accuracy(sigmoid>0.5라는 "모호한
# 구간 포함 전체 정확도")를 기준으로 했는데, 실제 매매 신호는 PROB_BUY_MIN(0.60)/
# PROB_SELL_MAX(0.40) 확신 구간에서만 나간다 — 즉 안전장치가 지키는 지표와 실제로
# 돈이 걸리는 지표가 서로 달라, 전체 정확도는 유지돼도 확신 구간 정확도만
# 나빠지는 경우 롤백이 전혀 트리거되지 않을 수 있었다 (analysis_team lstm 리포트
# 2026-11-22_2118 발견점 #1, synthesizer Top3 #2). 확신 구간만의 정확도
# (val_accuracy_confident, 아래 train_model 참고)를 계산해 가능하면 그 값으로
# 롤백을 비교한다. 다만 확신 구간 샘플이 이 값보다 적으면(val set이 작거나
# 이번 구간 시세가 유난히 모호했던 경우) 소수 샘플만으로 계산한 정확도는 노이즈로
# 크게 흔들려 불필요한 롤백을 유발할 수 있어, 그럴 때는 기존 val_accuracy로
# 폴백한다 (같은 리포트의 리스크/트레이드오프 항목에서 제안한 최소 샘플 가드).
MIN_CONFIDENT_VAL_SAMPLES = 30

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
                 lr: float = 1e-3, finetune_lr: float | None = None, seq_len: int = SEQ_LEN,
                 val_frac: float = 0.15, finetune: bool = True, history_days: int | None = None,
                 on_epoch=None) -> dict:
    """한 코인에 대해 LSTM을 학습(또는 파인튜닝)하고 모델 파일로 저장.

    finetune=True(기본): 기존 저장된 가중치가 있으면 그걸 불러와 이어서 학습
    (추가학습). 없으면 새로 초기화해서 학습(최초 학습과 동일).
    finetune=False: 항상 처음부터 새로 학습.

    finetune_lr: 기존 가중치를 이어 학습(파인튠)할 때 쓸 학습률. None(기본)이면
    lr/3을 자동으로 쓴다. 매 재학습(main_loop.py의 RETRAIN_TIMES_LOCAL 스케줄,
    하루 10회)이 거의 동일한 HISTORY_DAYS_FOR_TRAIN일 윈도우를 같은 lr로 반복
    학습하면, "오늘 새로 들어온 데이터"가 전체 학습 신호에서 차지하는 비중이
    구조적으로 작아서(윈도우 겹침 때문에) 가중치가 매번 과거 패턴을 다시
    확인하는 쪽에 더 가깝게 갱신된다. 파인튠 시 lr을 낮추면 기존에 학습된
    가중치가 매번 급격히 덮어씌워지는 것을 완화할 수 있다. 최초 학습
    (resumed_from_existing=False)에는 영향 없음 — 그때는 항상 lr을 그대로 쓴다.

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
    # build_dataset은 겹치는 슬라이딩 윈도우(seq_len분)로 샘플을 만들기 때문에,
    # split 경계 바로 앞 train 샘플과 바로 뒤 val 샘플은 입력 윈도우의 대부분을
    # 공유한다. train 쪽에서 seq_len개만큼 여유를 둬서 그 겹침을 없애면
    # val_accuracy가 실제보다 낙관적으로 보이는 걸 줄일 수 있다 (val 샘플 수는
    # 그대로 유지, train만 경계 근처 일부를 버림 — 전체 샘플 대비 영향은 작음).
    train_end = max(0, split - seq_len)
    X_train, X_val = X[:train_end], X[split:]
    y_train, y_val = y[:train_end], y[split:]

    model = LSTMClassifier().to(DEVICE)
    existing_path = MODEL_DIR / f"{coin}_lstm.pt"
    resumed_from_existing = finetune and existing_path.exists()
    if resumed_from_existing:
        # map_location=DEVICE: 예전에 CPU로 저장된 모델이든, 다른 디바이스에서
        # 저장된 모델이든 지금 실행 중인 디바이스(GPU/CPU)로 바로 올라오게 한다.
        # weights_only=True: 우리가 직접 저장한(cpu_state 딕셔너리) 파일만
        # 읽으므로 안전하게 켤 수 있고, pickle 전체 허용 경고도 사라진다.
        model.load_state_dict(torch.load(existing_path, map_location=DEVICE, weights_only=True))

    # 파인튠(이어학습)일 때만 더 낮은 학습률을 쓴다 — 최초 학습은 lr 그대로.
    effective_lr = (finetune_lr if finetune_lr is not None else lr / 3) if resumed_from_existing else lr
    opt = torch.optim.Adam(model.parameters(), lr=effective_lr)
    loss_fn = nn.BCEWithLogitsLoss()

    X_train_t = torch.from_numpy(X_train).to(DEVICE)
    y_train_t = torch.from_numpy(y_train).to(DEVICE)
    X_val_t = torch.from_numpy(X_val).to(DEVICE)
    y_val_t = torch.from_numpy(y_val).to(DEVICE)

    model.train()
    n_batches = max(1, len(X_train) // batch_size)
    for epoch in range(epochs):
        perm = torch.randperm(len(X_train), device=DEVICE)
        for b in range(n_batches):
            idx = perm[b * batch_size: (b + 1) * batch_size]
            if len(idx) == 0:
                continue
            opt.zero_grad()
            logits = model(X_train_t[idx])
            loss = loss_fn(logits, y_train_t[idx])
            loss.backward()
            opt.step()
        if on_epoch:
            # 데이터량이 커지면(90일치) 코인당 학습이 오래 걸릴 수 있어,
            # run.py(구 watchdog)의 하트비트가 코인 단위로만 찍히면 간극이
            # 너무 벌어질 수 있어 epoch 단위로도 찍을 수 있게 콜백을 열어둔다.
            on_epoch(coin, epoch + 1, epochs)

    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t)
        val_probs = torch.sigmoid(val_logits)
        val_pred = (val_probs > 0.5).float()
        val_acc = (val_pred == y_val_t).float().mean().item()

        # 실제 매매 신호는 PROB_BUY_MIN/PROB_SELL_MAX 확신 구간에서만 나가므로,
        # 그 구간만의 정확도도 따로 계산해둔다 (위 MIN_CONFIDENT_VAL_SAMPLES
        # 설명 참고). 샘플이 가드값보다 적으면 노이즈가 커서 None으로 둔다.
        confident_mask = (val_probs >= PROB_BUY_MIN) | (val_probs <= PROB_SELL_MAX)
        n_confident_val_samples = int(confident_mask.sum().item())
        val_accuracy_confident = None
        if n_confident_val_samples >= MIN_CONFIDENT_VAL_SAMPLES:
            val_accuracy_confident = (
                val_pred[confident_mask] == y_val_t[confident_mask]
            ).float().mean().item()

    # 이전 저장 모델의 val_accuracy와 비교해, 이번 학습으로 성능이 크게
    # 떨어졌으면 저장을 건너뛰고 기존 가중치를 그대로 유지한다 (아래
    # VAL_ACC_ROLLBACK_THRESHOLD 설명 참고). 비교 기준은 "이번에 가중치를
    # 이어받았는지(resumed_from_existing)"가 아니라 "비교할 이전 모델 파일이
    # 있는지(existing_path.exists())"다 — finetune_chain_length가
    # FINETUNE_CHAIN_LENGTH_RESET_THRESHOLD를 넘어 main_loop.py가 finetune=False로
    # 전체 재학습을 강제하는 경우에도 existing_path는 이미 존재하므로, 그렇게
    # 처음부터 다시 학습한 모델도 직전 저장 모델보다 val_acc가 크게 떨어지면
    # 롤백돼야 한다. resumed_from_existing만 보면 이 경로에서 롤백 비교가
    # 통째로 스킵돼, 안전장치 없이 성능이 나쁜 모델이 그대로 저장·추론에
    # 쓰이는 공백이 있었다 (analysis_team 리포트 2026-11-22_1326_lstm.md,
    # 2026-11-22_1418_lstm.md에서 지적). 완전히 처음 학습하는 코인(모델 파일
    # 자체가 없음)만 비교 대상이 없으므로 항상 저장한다.
    meta_path = MODEL_DIR / f"{coin}_lstm_meta.json"
    prev_val_acc = None
    prev_val_acc_confident = None
    # "영구 이어학습(finetune=True 기본값)" 구조에서는 전체 재학습(finetune=False)이
    # 언제 한 번 일어났는지 호출부(main_loop.py)가 알 방법이 없었다 — 이 파일
    # 안에는 그런 리셋을 스스로 트리거하는 로직도 없다 (analysis_team 리포트
    # 2026-11-21_2118_lstm.md 제안). finetune_chain_length를 meta에 남겨서,
    # 연속 파인튠 횟수를 추적할 수 있게 한다. 학습 자체(가중치/lr/저장 로직)는
    # 전혀 바꾸지 않고, 관측 가능한 메타데이터만 추가하는 변경이다.
    prev_chain_length = 0
    if existing_path.exists() and meta_path.exists():
        try:
            prev_meta = json.loads(meta_path.read_text())
            prev_val_acc = prev_meta.get("val_accuracy")
            prev_val_acc_confident = prev_meta.get("val_accuracy_confident")
            prev_chain_length = prev_meta.get("finetune_chain_length", 0)
        except (json.JSONDecodeError, OSError):
            prev_val_acc = None
            prev_val_acc_confident = None
            prev_chain_length = 0

    # 비교 기준: 이전/이번 둘 다 확신 구간 정확도가 충분한 샘플로 계산돼 있으면
    # (val_accuracy_confident 둘 다 not None) 그 값으로 롤백을 비교한다 — 실제
    # 매매 임계값 기준의 정확도가 안전장치의 기준이 되어야 하기 때문이다
    # (analysis_team lstm 리포트 2026-11-22_2118 발견점 #1, synthesizer Top3 #2).
    # 둘 중 하나라도 없으면(샘플 부족, 또는 이 필드가 아직 없던 과거 모델과
    # 비교하는 경우) 기존 방식(val_acc, 모호한 구간 포함 전체 정확도)으로
    # 폴백한다 — 기존 동작을 그대로 유지하는 안전한 기본값.
    if prev_val_acc_confident is not None and val_accuracy_confident is not None:
        _compare_label = "val_accuracy_confident"
        _prev_compare, _curr_compare = prev_val_acc_confident, val_accuracy_confident
    else:
        _compare_label = "val_accuracy"
        _prev_compare, _curr_compare = prev_val_acc, val_acc

    should_save = True
    skip_reason = None
    if existing_path.exists() and _prev_compare is not None and (_prev_compare - _curr_compare) > VAL_ACC_ROLLBACK_THRESHOLD:
        should_save = False
        skip_reason = (f"{_compare_label} 하락으로 롤백: 이전 {_prev_compare:.4f} -> 이번 {_curr_compare:.4f} "
                        f"(임계값 {VAL_ACC_ROLLBACK_THRESHOLD})")
        log.warning(f"[{coin}] {skip_reason} — 기존 가중치 유지, 저장 생략")

    # 이번 학습이 파인튠(이어학습)이면 체인을 1 늘리고, 전체 재학습(처음부터
    # 새로 초기화해서 학습)이면 0으로 리셋한다. 몇 회 연속 파인튠됐는지가
    # 쌓이므로, 나중에 사람이 "N회 이상이면 전체 재학습" 같은 정책을 메타만
    # 보고 판단할 수 있다 — 이 변경 자체는 그런 정책을 자동으로 적용하지 않는다.
    finetune_chain_length = (prev_chain_length + 1) if resumed_from_existing else 0

    if should_save:
        # 저장은 항상 CPU 텐서로 — GPU 없는 다른 PC/환경에서도 그대로
        # load_state_dict가 가능하게 디바이스에 종속되지 않는 파일로 남긴다.
        cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(cpu_state, existing_path)
        date_str = pd.Timestamp.now().strftime("%Y-%m-%d")
        torch.save(cpu_state, HISTORY_DIR / f"{coin}_lstm_{date_str}.pt")
        meta_path.write_text(json.dumps({
            "val_accuracy": val_acc,
            "val_accuracy_confident": val_accuracy_confident,
            "n_confident_val_samples": n_confident_val_samples,
            "timestamp": timestamp,
            "finetune_chain_length": finetune_chain_length,
        }))

        _loaded_models.pop(coin, None)  # 추론 캐시 무효화 (다음 predict_proba가 새 가중치 로드)

    return {
        "coin": coin,
        "trained": True,
        "saved": should_save,
        "reason": skip_reason or "",
        "finetuned": resumed_from_existing,
        "lr": lr,
        "effective_lr": effective_lr,
        "n_samples": n,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "val_accuracy": round(val_acc, 4),
        "val_accuracy_confident": round(val_accuracy_confident, 4) if val_accuracy_confident is not None else None,
        "n_confident_val_samples": n_confident_val_samples,
        "prev_val_accuracy": prev_val_acc,
        "prev_val_accuracy_confident": prev_val_acc_confident,
        "finetune_chain_length": finetune_chain_length if should_save else prev_chain_length,
        "epochs": epochs,
        "timestamp": timestamp,
        "data_start": data_start,
        "data_end": data_end,
        "history_days": history_days,
        "seq_len": seq_len,
    }


def train_all(candles: dict[str, pd.DataFrame], **kwargs) -> list[dict]:
    return [train_model(coin, df, **kwargs) for coin, df in candles.items()]


def get_finetune_chain_length(coin: str) -> int:
    """{coin}_lstm_meta.json에 저장된 현재 finetune_chain_length(연속 파인튠 횟수)를
    조회. 파일이 없거나 읽기 실패 시 0("아직 파인튠 체인 없음"과 동일하게 안전한
    기본값)을 반환한다.

    train_model()은 이미 이 값을 meta.json에 기록만 하고 자동으로 쓰지는
    않았다(위 finetune_chain_length 변수와 주석 참고) — "영구 이어학습" 구조에서
    언제 한 번 전체 재학습(finetune=False)이 필요한지 메타만 보고 판단할 수
    있게 하기 위한 조회용 헬퍼다. 학습 로직(가중치/lr/저장) 자체는 전혀 건드리지
    않는다 (analysis_team lstm 리포트 2026-11-22_0018, synthesizer Top3 #2 — 호출부는
    main_loop.py의 retrain_lstm())."""
    meta_path = MODEL_DIR / f"{coin}_lstm_meta.json"
    if not meta_path.exists():
        return 0
    try:
        meta = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return 0
    return meta.get("finetune_chain_length") or 0


_loaded_models: dict[str, LSTMClassifier] = {}


def _load_model(coin: str) -> LSTMClassifier | None:
    """추론용 모델은 의도적으로 항상 CPU에 둔다. predict_proba는 매분,
    시퀀스 1개짜리 forward pass만 하는 가벼운 연산이라, 매번 CPU<->GPU로
    데이터를 옮기는 비용이 계산 자체보다 더 클 수 있다. 학습(train_model)만
    DEVICE(GPU 있으면 GPU)를 쓰고, 저장은 항상 CPU 텐서로 통일해둔 덕분에
    여기서는 추가 변환 없이 바로 CPU로 로드할 수 있다."""
    path = MODEL_DIR / f"{coin}_lstm.pt"
    if not path.exists():
        return None
    if coin not in _loaded_models:
        model = LSTMClassifier()
        # weights_only=True: 우리가 직접 저장한 파일만 읽으므로 안전하게
        # 켤 수 있고, pickle 전체 허용 경고도 사라진다.
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
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
        # signal()/signal_with_confidence()가 마지막으로 계산한 prob_up을
        # 캐싱해둔다. main_loop.py가 매 사이클 같은 df로 예측 기록 저장용
        # predict_proba를 또 호출해 특징생성+forward pass가 코인당 두 번
        # 실행되던 중복을 없애기 위함 (analysis_team execution 리포트
        # 2026-11-22_1418/1326, synthesizer Top3 #3). signal_with_confidence가
        # 코인 하나를 처리한 직후(main_loop.py의 같은 코인 루프 반복 안에서)
        # 바로 읽어가는 용도이므로, 다음 코인으로 넘어가 다시 쓰이기 전까지만
        # 유효하다고 가정한다.
        self.last_prob_up: float | None = None

    def min_bars(self) -> int:
        return SEQ_LEN + 1

    def signal(self, df: pd.DataFrame) -> str:
        if len(df) < self.min_bars():
            return "HOLD"
        prob_up = predict_proba(self.coin, df)
        self.last_prob_up = prob_up
        if prob_up is None:
            return "HOLD"  # 아직 학습된 모델 없음
        if prob_up >= PROB_BUY_MIN:
            return "BUY"
        if prob_up <= PROB_SELL_MAX:
            return "SELL"
        return "HOLD"

    def signal_with_confidence(self, df: pd.DataFrame) -> tuple[str, float]:
        """prob_up이 0.5(완전 모호)에서 얼마나 멀리 떨어져 있는지를 확신도로
        쓴다. 임계값(PROB_BUY_MIN/PROB_SELL_MAX)을 막 넘은 경우는 확신도가
        낮고, 0.9나 0.1처럼 극단으로 갈수록 1.0에 가까워진다. main_loop.py가
        이 값으로 베팅 크기를 정한다."""
        if len(df) < self.min_bars():
            return "HOLD", 0.0
        prob_up = predict_proba(self.coin, df)
        self.last_prob_up = prob_up
        if prob_up is None:
            return "HOLD", 0.0
        if prob_up >= PROB_BUY_MIN:
            sig = "BUY"
        elif prob_up <= PROB_SELL_MAX:
            sig = "SELL"
        else:
            return "HOLD", 0.0
        # 임계값(0.5에서 threshold_distance만큼 떨어진 지점)부터 극단(0.5만큼
        # 떨어진 지점, 즉 prob_up=1.0 또는 0.0)까지를 0~1로 선형 스케일.
        distance_from_mid = abs(prob_up - 0.5)
        threshold_distance = PROB_BUY_MIN - 0.5  # = 0.10 (PROB_BUY_MIN=0.60 기준)
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
