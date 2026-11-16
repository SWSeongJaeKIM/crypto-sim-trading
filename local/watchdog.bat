@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM Watchdog for main_loop.py.
REM
REM main_loop.py cannot detect its own hang (once it is truly stuck, the
REM code inside it stops running, so it can't restart itself). This script
REM runs outside it, checks heartbeat.txt's last-modified time, and force
REM restarts main_loop.py if that file hasn't moved in STALE_MINUTES.
REM Also restarts if the main_loop.py window has simply disappeared (crash).
REM
REM Usage: double-click this file, or run it from cmd.
REM (Run this instead of running "python main_loop.py" directly.)
REM To stop: close this window (also close the main_loop window it opened).
REM ============================================================

set HEARTBEAT_FILE=%~dp0heartbeat.txt
set STALE_MINUTES=5
set CHECK_INTERVAL_SEC=30

echo [watchdog] Watching main_loop.py (restart if heartbeat stale for %STALE_MINUTES% min)
echo [watchdog] Heartbeat file: %HEARTBEAT_FILE%

:restart_loop
REM In case a previous run left a window behind, clean it up before
REM starting a new one (/t = kill the whole process tree).
taskkill /f /t /fi "windowtitle eq crypto-sim main_loop*" >nul 2>&1

echo.
echo [watchdog] %date% %time% - starting main_loop.py
start "crypto-sim main_loop" /min cmd /c "python main_loop.py"

REM Give it time to start up before the first check (heartbeat.txt may not
REM exist yet, or may hold a stale value from a previous run).
timeout /t 60 /nobreak >nul

:watch_loop
timeout /t %CHECK_INTERVAL_SEC% /nobreak >nul

REM If the "crypto-sim main_loop" window is gone entirely (crashed), restart.
tasklist /fi "windowtitle eq crypto-sim main_loop*" /fo csv | findstr /i "cmd.exe" >nul
if errorlevel 1 (
    echo [watchdog] %date% %time% - main_loop window is gone, restarting
    goto restart_loop
)

if not exist "%HEARTBEAT_FILE%" (
    echo [watchdog] %date% %time% - heartbeat file not found yet, waiting
    goto watch_loop
)

REM Compute minutes since heartbeat.txt was last modified via PowerShell
REM (batch alone is awkward for time arithmetic).
for /f %%M in ('powershell -NoProfile -Command "[math]::Floor(((Get-Date) - (Get-Item '%HEARTBEAT_FILE%').LastWriteTime).TotalMinutes)"') do set AGE_MIN=%%M

if %AGE_MIN% GEQ %STALE_MINUTES% (
    echo [watchdog] %date% %time% - heartbeat stale for %AGE_MIN% min ^(threshold %STALE_MINUTES%^), forcing restart
    taskkill /f /t /fi "windowtitle eq crypto-sim main_loop*" >nul 2>&1
    timeout /t 3 /nobreak >nul
    goto restart_loop
)

goto watch_loop
