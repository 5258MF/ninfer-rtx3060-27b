@echo off
chcp 65001 >nul
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0launcher\launch.ps1" %*
set "IQ3_RC=%ERRORLEVEL%"
if /I "%~1"=="dryrun" exit /b %IQ3_RC%
if /I "%~1"=="wizard-dryrun" exit /b %IQ3_RC%
pause
exit /b %IQ3_RC%
