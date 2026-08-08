@echo off
title Aperiodic Percolation Portal
cd /d "%~dp0"

REM Use 'python' if it's on PATH, otherwise fall back to the 'py' launcher.
set "PY=python"
where python >nul 2>&1 || set "PY=py"

echo ============================================================
echo    Aperiodic Percolation Portal
echo ============================================================
echo Checking dependencies...
%PY% -m streamlit version >nul 2>&1
if errorlevel 1 (
    echo Streamlit not found - installing it now, one moment...
    %PY% -m pip install streamlit
)
echo.
echo Starting the portal - a browser tab will open automatically.
echo Keep THIS window open while you use it; close it to stop the portal.
echo.
%PY% -m streamlit run interface/gui_app.py

echo.
echo The portal has stopped.
pause
