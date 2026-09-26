@echo off
setlocal
cd /d "%~dp0"
py -3.12 -c "import sys; print(sys.version)"
if errorlevel 1 goto fail
if not exist ".venv\Scripts\python.exe" py -3.12 -m venv .venv
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install -r requirements-local.txt
if errorlevel 1 goto fail
".venv\Scripts\python.exe" run_local.py --demo --check
if errorlevel 1 goto fail
echo Setup complete. Run start_demo.cmd for local preview.
pause
exit /b 0
:fail
echo Setup failed. See the message above and TEAM_SETUP.md.
pause
exit /b 1
