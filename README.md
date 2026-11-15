# crypto-sim-trading

6대 암호화폐(BTC, ETH, XRP, SOL, DOGE, ADA) 실시세 기반 모의투자 시스템.
조건식(규칙 기반) 전략 4종이 경쟁하며, 성과가 저조한 전략은 과거 데이터
백테스트로 파라미터를 자동 재조정합니다.

## 구성

- `strategies.py` — 4개 조건식 전략 (이동평균교차+RSI, 볼린저밴드+RSI, MACD 모멘텀, 추세추종)
- `portfolio.py` — 가상 계좌 모델 (전략별 독립 계좌, 코인별 보유 슬롯)
- `learner.py` — 전략 파라미터 자동 재조정 (랜덤 서치 + 백테스트)
- `data_fetcher.py` — Upbit 공개 API로 실시간/전체 역사 시세 수집
- `engine.py` — 사이클 실행 (순수 함수: 상태 dict → 갱신된 상태 dict)

## 아키텍처

모든 모듈이 DB/파일 I/O 없는 순수 함수로 작성되어 있습니다. 상태는
Claude Artifact DB에 저장되며, 매시간 스케줄 작업이 이 저장소를 clone해서
`engine.run_cycle()`을 실행 → DB를 갱신하는 방식으로 동작합니다
(스케줄 작업은 매번 새 컨테이너에서 시작되어 로컬 파일시스템이 공유되지 않기 때문).

대시보드: Claude Artifact (slug `987f46cc-431a-4d00-805b-1f09aeea3ab8`)
