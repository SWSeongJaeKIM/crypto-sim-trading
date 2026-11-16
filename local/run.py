"""
main_loop.py를 감시하는 실행 스크립트. main_loop.py를 직접 실행하는 대신
이 파일을 실행하면 된다:

  python3 run.py

main_loop.py가 자기 자신이 멈췄다는 걸 알 방법은 없다(진짜로 멈추면
그 안의 코드 자체가 더 이상 실행되지 않으므로). 그래서 이 워치독을
"별도 프로세스"로 띄워서, main_loop.py를 자식 프로세스로 실행시켜두고
바깥에서 지켜본다. 다음 두 경우 자동으로 강제 종료 후 재시작한다:

  1. main_loop.py 프로세스가 그냥 죽은 경우 (예외로 꺼짐 등)
  2. 프로세스는 살아있지만 멈춘(행) 경우 — main_loop.py가 매 사이클/
     학습 단계마다 heartbeat.txt에 "방금 살아있었다"는 시각을 기록하는데,
     이 파일이 STALE_SECONDS 이상 안 바뀌면 "멈췄다"고 판단한다.

이전에는 .bat 스크립트로 같은 역할을 시도했으나, 한국어 Windows의
cmd.exe가 UTF-8로 쓰인 .bat 파일을 시스템 코드페이지(CP949)로 잘못
해석해서 taskkill/tasklist 같은 명령어 토큰 자체가 깨지는 문제가 있었다.
Python은 플랫폼에 상관없이 동일하게 동작하므로 이 문제가 없다.

중지하려면 Ctrl+C (워치독과 그게 띄운 main_loop.py 자식 프로세스를
함께 정리하고 종료한다).
"""

from __future__ import annotations

import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
HEARTBEAT_PATH = HERE / "heartbeat.txt"
MAIN_LOOP_SCRIPT = HERE / "main_loop.py"

STALE_SECONDS = 5 * 60       # 하트비트가 이만큼 안 바뀌면 "멈춤"으로 판단
CHECK_INTERVAL_SEC = 15      # 이 주기로 하트비트/프로세스 상태 확인
STARTUP_GRACE_SEC = 60       # 막 시작한 뒤에는 하트비트가 없거나 오래된 상태일 수 있으니 유예

_stop = False


def _log(msg: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[watchdog {now}] {msg}", flush=True)


def _handle_sigint(signum, frame):
    global _stop
    _log("종료 신호 수신 — main_loop.py 정리하고 워치독을 끝냅니다.")
    _stop = True


signal.signal(signal.SIGINT, _handle_sigint)
signal.signal(signal.SIGTERM, _handle_sigint)


def _heartbeat_age_sec() -> float | None:
    """heartbeat.txt의 마지막 수정 이후 경과 시간(초). 파일이 없으면 None."""
    try:
        mtime = HEARTBEAT_PATH.stat().st_mtime
    except FileNotFoundError:
        return None
    return time.time() - mtime


def _start_main_loop() -> subprocess.Popen:
    _log(f"main_loop.py 시작: {MAIN_LOOP_SCRIPT}")
    # 오래된 하트비트가 "방금 막 쓴 것"처럼 보여서 바로 재시작 루프에 빠지는
    # 걸 막기 위해, 시작하기 전에 기존 하트비트 파일을 지운다(있다면).
    try:
        HEARTBEAT_PATH.unlink()
    except FileNotFoundError:
        pass
    return subprocess.Popen(
        [sys.executable, str(MAIN_LOOP_SCRIPT)],
        cwd=str(HERE),
    )


def _terminate(proc: subprocess.Popen, reason: str) -> None:
    _log(f"main_loop.py 강제 종료 ({reason})")
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        _log("정상 종료 안 돼서 kill()로 강제 종료")
        proc.kill()
        proc.wait(timeout=10)


def main() -> None:
    _log(f"감시 시작 (하트비트 {STALE_SECONDS}초 이상 끊기면 재시작, {CHECK_INTERVAL_SEC}초마다 확인)")
    _log(f"하트비트 파일: {HEARTBEAT_PATH}")

    while not _stop:
        proc = _start_main_loop()
        started_at = time.time()

        while not _stop:
            time.sleep(CHECK_INTERVAL_SEC)

            # 1) 프로세스가 그냥 죽었는지 (exit code가 생겼으면 종료된 것)
            exit_code = proc.poll()
            if exit_code is not None:
                _log(f"main_loop.py 프로세스가 종료됨 (exit code {exit_code}) — 재시작합니다")
                break

            # 2) 막 시작한 직후면 유예 기간 동안은 하트비트 안 봄
            if time.time() - started_at < STARTUP_GRACE_SEC:
                continue

            # 3) 하트비트가 너무 오래 안 바뀌었는지 (멈춤/행 감지)
            age = _heartbeat_age_sec()
            if age is None:
                _log("하트비트 파일이 아직 없음 — 계속 대기")
                continue
            if age >= STALE_SECONDS:
                _log(f"하트비트가 {age:.0f}초째 안 바뀜 (기준 {STALE_SECONDS}초) — 멈춘 것으로 판단")
                _terminate(proc, "하트비트 멈춤")
                break

        if _stop:
            if proc.poll() is None:
                _terminate(proc, "워치독 종료 요청")
            break

        time.sleep(3)  # 재시작 전 잠깐 숨 고르기

    _log("워치독 종료됨")


if __name__ == "__main__":
    main()
