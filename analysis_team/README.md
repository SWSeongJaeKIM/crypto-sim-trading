# analysis_team/ — 6에이전트 코드 분석팀

**이 폴더는 트레이딩 시스템과 완전히 별개**다. `local/`의 `main_loop.py`,
`dashboard.py` 등은 사용자 PC에서 상시 돌며 실제 모의투자를 하고, 여기
`analysis_team/`은 클라우드에서 1시간마다 한 번씩 GitHub 저장소의 **코드
자체를 리뷰**하고 개선 방향을 제안하는 별도 작업이다.

- Claude API 사용량이 든다 (사용자가 명시적으로 괜찮다고 확인함).
- PC의 `trading.db`(실제 운영 데이터: 거래내역, 손익, 학습기록 등)에는
  접근하지 않는다 — 클라우드 스케줄 작업이 사용자 PC에 직접 붙을 수 없기
  때문. 그래서 **코드 구조/로직만 보고** 분석한다("이 조건식은 이렇게
  개선하면 어떨까", "이 부분은 버그 소지가 있다" 같은 리뷰).
- 결과는 `analysis_team/reports/<타임스탬프>.md` 파일로 저장되어 GitHub에
  커밋된다. 로컬 `dashboard.py`가 GitHub에서 최신 리포트를 읽어와 "AI
  분석팀" 탭에 보여준다.

## 6개 역할

매 실행마다 아래 6명이 각자 맡은 영역만 보고 독립적으로 리포트를 쓴다.
서로 다른 관심사로 나눠서 중복 제안을 줄였다.

1. **전략 분석가** (`strategy`) — `local/strategies.py`의 조건식 4종
   (ma_cross_rsi, bollinger_rsi, macd_momentum, trend_follow). 진입/청산
   조건, 파라미터 기본값, 로직상 허점을 점검.
2. **LSTM 분석가** (`lstm`) — `local/lstm_model.py`. 피처 구성, 시퀀스
   길이, 학습 방식(파인튜닝), 재학습 주기, 데이터 양을 점검.
3. **실행/버그 분석가** (`execution`) — `local/main_loop.py`,
   `local/data_fetcher.py`. 예외처리, 사이클 타이밍, API 호출 안정성,
   잠재적 버그.
4. **데이터/저장소 분석가** (`storage`) — `local/storage.py`. 스키마
   설계, 쿼리 효율, 데이터 정합성(중복/누락 가능성).
5. **리스크 분석가** (`risk`) — `local/portfolio.py` + 매매 판단 쪽.
   포지션 크기(TRADE_FRACTION), 리셋/누적 구조, 자금 분산, 특정 코인
   편중 위험.
6. **종합 조정자** (`synthesizer`) — 위 5명의 리포트를 읽고 중복을
   제거하고 "지금 가장 먼저 해볼 만한 개선 Top 3"로 정리. 이 역할은
   나머지 5개가 먼저 끝난 뒤에 실행해야 한다.

## 파일

- `roles.md` — 각 역할에게 줄 구체적인 지시사항(프롬프트) 정의
- `reports/` — 역할별 리포트가 쌓이는 곳. 파일명 규칙:
  `reports/<YYYY-MM-DD_HHmm>_<role>.md`
- `reports/latest.json` — 대시보드가 바로 읽을 수 있도록, 각 역할의
  최신 리포트 경로와 한 줄 요약을 모아둔 인덱스. 매 실행마다 갱신.
