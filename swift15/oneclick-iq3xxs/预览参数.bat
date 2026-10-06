@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0launcher\launch.ps1" -Mode dryrun
pause
