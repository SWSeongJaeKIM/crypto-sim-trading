@echo off
setlocal enabledelayedexpansion

:: ============================================================
:: main_loop.py 워치독.
::
:: main_loop.py는 "자기가 멈췄다"는 걸 스스로 알 방법이 없다(멈추면
:: 그 안의 코드 자체가 더 이상 실행되지 않으므로). 그래서 바깥에서 도는
:: 이 .bat가 heartbeat.txt 파일의 "최종 수정 시각"을 주기적으로 확인해서,
:: 너무 오래 안 바뀌었으면 "멈췄다"고 판단하고 강제로 끝낸 뒤 다시 켠다.
:: main_loop.py가 그냥 죽어서 python 프로세스 자체가 없어진 경우도
:: 똑같이 다시 켠다.
::
:: 사용법: 이 파일을 더블클릭하거나 cmd에서 watchdog.bat 실행.
:: (python main_loop.py를 직접 실행하는 대신 이걸 실행하면 됨)
:: 끄려면 이 창을 닫거나 Ctrl+C (연 python 창도 따로 닫아야 완전 종료).
:: ============================================================

set HEARTBEAT_FILE=%~dp0heartbeat.txt
set STALE_MINUTES=5
set CHECK_INTERVAL_SEC=30

echo [watchdog] main_loop.py 감시 시작 (하트비트 %STALE_MINUTES%분 이상 끊기면 재시작)
echo [watchdog] 하트비트 파일: %HEARTBEAT_FILE%

:restart_loop
:: 혹시 이전 회차가 죽다 말아서 python.exe가 남아있을 수 있으니, 새로
:: 띄우기 전에 같은 창 제목을 가진 프로세스 트리를 먼저 정리한다.
:: (/t = 자식 프로세스까지 트리 전체 종료. 창 제목이 없으면 그냥 넘어감)
taskkill /f /t /fi "windowtitle eq crypto-sim main_loop*" >nul 2>&1

echo.
echo [watchdog] %date% %time% - main_loop.py 시작
start "crypto-sim main_loop" /min cmd /c "python main_loop.py"

:: main_loop.py가 막 시작한 직후엔 heartbeat.txt가 아직 없거나 오래된
:: 값일 수 있으니, 첫 체크까지 넉넉히 기다려준다.
timeout /t 60 /nobreak >nul

:watch_loop
timeout /t %CHECK_INTERVAL_SEC% /nobreak >nul

:: "crypto-sim main_loop" 창 제목을 가진 프로세스가 아예 없어졌으면
:: (죽었으면) 바로 재시작. 창 제목으로 필터링해서 사용자가 띄워둔 다른
:: python.exe 프로그램과 섞이지 않게 한다.
tasklist /fi "windowtitle eq crypto-sim main_loop*" /fo csv | findstr /i "cmd.exe" >nul
if errorlevel 1 (
    echo [watchdog] %date% %time% - main_loop.py 창이 사라짐, 재시작합니다
    goto restart_loop
)

if not exist "%HEARTBEAT_FILE%" (
    echo [watchdog] %date% %time% - 하트비트 파일이 아직 없음, 계속 대기
    goto watch_loop
)

:: PowerShell로 "하트비트 파일이 마지막으로 바뀐 지 몇 분 지났는지" 계산.
:: (배치 파일만으로는 시간 뺄셈이 번거로워 한 줄만 PowerShell에 위임)
for /f %%M in ('powershell -NoProfile -Command "[math]::Floor(((Get-Date) - (Get-Item '%HEARTBEAT_FILE%').LastWriteTime).TotalMinutes)"') do set AGE_MIN=%%M

if %AGE_MIN% GEQ %STALE_MINUTES% (
    echo [watchdog] %date% %time% - 하트비트가 %AGE_MIN%분째 안 바뀜 (%STALE_MINUTES%분 기준 초과) - 멈춘 것으로 판단, 강제 재시작
    taskkill /f /t /fi "windowtitle eq crypto-sim main_loop*" >nul 2>&1
    timeout /t 3 /nobreak >nul
    goto restart_loop
)

goto watch_loop
