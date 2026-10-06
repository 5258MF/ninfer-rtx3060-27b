@echo off
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0launcher\launch.ps1" %*
set "NINFER_RC=%ERRORLEVEL%"
echo.
pause
exit /b %NINFER_RC%
