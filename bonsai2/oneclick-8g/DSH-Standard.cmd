@echo off
where py >nul 2>&1
if errorlevel 1 (
  python "%~dp0launcher\dsh-client.py" standard %*
) else (
  py -3 "%~dp0launcher\dsh-client.py" standard %*
)
if errorlevel 1 pause
