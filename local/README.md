# local/ — PC 상주 실행용 모의투자 시스템

**Claude나 그 어떤 외부 AI API도 호출하지 않습니다.** 이 폴더의 코드만으로
사용자 PC에서 완전히 독립적으로 돈다. 사용량이 드는 건 Upbit 공개 API
호출(무료)뿐이다.

## 구성

- `strategies.py` — 조건식 전략 4종 (상위 폴더와 동일, 복사본)
- `portfolio.py` — 가상 계좌 모델 (상위 폴더와 동일, 복사본)
- `learner.py` — 전략 파라미터 자동 재조정 (상위 폴더와 동일, 복사본)
- `data_fetcher.py` — Upbit 1분봉 수집 (실시간 + 학습용 N일 히스토리)
- `lstm_model.py` — **LSTM 분류 모델**: 최근 60분 흐름(수익률, 거래량 변화,
  RSI, MACD, 볼린저 %b)을 보고 "다음 1분봉이 오를지/내릴지" 추론.
  조건식 전략과 동일한 `signal(df)` 인터페이스(`LSTMStrategy`)로 노출되어
  같은 리더보드에서 경쟁한다.
- `storage.py` — 로컬 SQLite(`trading.db`) 저장소 (포트폴리오/거래/스냅샷/파라미터/
  LSTM 학습기록)
- `main_loop.py` — **상주 실행 스크립트**. 매 1분 5개 전략(조건식4+LSTM1)이
  신호 계산 → 모의 매수/매도 → SQLite 저장을 반복. 매일 새벽 4시(로컬시간)에
  LSTM을 최근 14일 데이터로 **추가학습(파인튜닝)**. 지금 뭘 하고 있는지
  (대기중/데이터수집중/학습중 등)를 매 단계마다 SQLite에 기록해서 대시보드에서
  볼 수 있게 한다.
- `dashboard.py` — 로컬 Flask 웹서버. 전략별 수익률 리더보드, 지금 진행 상황
  표시줄, 최근 거래, **LSTM 학습 기록(코인별 정확도 추이)**을 보여준다.

## LSTM 학습 방식 (추가학습)

매일 재학습할 때마다 모델을 처음부터 새로 만들지 않고, **기존에 저장된
가중치를 불러와서 이어서 학습(파인튜닝)**한다. 그래서:

- 예전에 학습한 패턴을 급격히 잃지 않으면서 최근 데이터를 반영
- 학습할 때마다 `model_trainings` 테이블에 기록이 **누적**된다 (덮어쓰지 않음)
  — 언제, 어떤 코인, 최초학습/추가학습 여부, 데이터 기간, 샘플 수, 검증
  정확도를 전부 남겨서 대시보드에서 정확도가 올라가는지/내려가는지 추이를 볼 수 있다.
- 모델 가중치 파일도 두 군데 저장: `models/{coin}_lstm.pt`(최신, 실제
  추론에 쓰는 파일, 매번 덮어씀)와 `models/history/{coin}_lstm_YYYY-MM-DD.pt`
  (그날짜 사본, 나중에 특정 날짜 모델로 되돌리거나 비교하고 싶을 때 사용).

## 설치 (최초 1회)

```bash
# Python 3.10+ 필요. 가상환경 권장.
cd local
pip install -r requirements.txt
```

GPU 없어도 된다 (CPU로 충분 — LSTM이 아주 작은 모델이라 1분 추론은 수 ms,
하루 1회 재학습도 노트북 기준 몇 분 내).

## 실행

터미널 2개를 띄운다 (또는 백그라운드로):

```bash
# 터미널 1: 매매 엔진 (계속 실행)
python3 main_loop.py

# 터미널 2: 대시보드
python3 dashboard.py
```

- 첫 실행 시 LSTM 모델이 없으므로 `main_loop.py`가 최근 14일치 1분봉을
  받아서 자동으로 최초 학습을 수행한다 (코인당 1~2분, 네트워크 속도에 따라
  다름 — 1분봉 14일치는 페이지네이션으로 받아야 해서 다소 걸림).
- 이후로는 매분 사이클이 돈다. Ctrl+C로 안전하게 종료 가능 (현재 사이클
  끝나고 정지).
- 대시보드: `http://localhost:5000`. 같은 와이파이의 폰에서 보려면
  PC의 사설 IP:5000 (Windows `ipconfig`, Mac `ipconfig getifaddr en0`로 확인).

## 24시간 계속 돌리기

회사에 가 있어도 집 PC가 계속 돌아야 하므로:

- **Windows**: 작업 스케줄러로 로그온 시 `main_loop.py`, `dashboard.py`를
  자동 시작하도록 등록 (또는 그냥 터미널 창을 열어두고 PC를 절전모드 안
  들어가게 설정).
- **Mac/Linux**: `nohup python3 main_loop.py &` 로 백그라운드 실행하거나,
  `launchd`(Mac)/`systemd`(Linux) 서비스로 등록.
- PC가 절전모드로 들어가면 스크립트도 멈추니, 전원 설정에서 "절전 모드
  안 함"으로 바꿔두는 게 중요하다.

## 코드 갱신 받기

GitHub 저장소(`SWSeongJaeKIM/crypto-sim-trading`)의 `local/` 폴더가 최신
코드다. 수정사항이 있으면:

```bash
git pull
```

실행 중이던 `main_loop.py`는 재시작해야 반영된다 (Ctrl+C 후 다시 실행).

## 데이터는 어디 저장되나요

전부 로컬 파일:
- `trading.db` — SQLite (포트폴리오, 거래기록, 스냅샷, LSTM 학습기록)
- `models/*.pt` — 코인별 최신 LSTM 가중치 (추론에 쓰는 파일)
- `models/history/*_YYYY-MM-DD.pt` — 날짜별 모델 가중치 백업
- `data_cache/*.csv` — 학습용으로 받은 1분봉 캐시
- `run.log` — 실행 로그

전부 `.gitignore`에 포함되어 있어 git에는 올라가지 않는다 (PC마다 자기
데이터로 새로 쌓임).
