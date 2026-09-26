@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
 echo Run setup_local.cmd first. Python 3.12 is required.
 pause
 exit /b 1
)
".venv\Scripts\python.exe" run_local.py --demo %*
pause
