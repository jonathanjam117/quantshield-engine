@echo off
title QuantShield Engine - +EV Betting System
echo ========================================================
echo           QUANT SHIELD BETTING ENGINE
echo       (Modo Seguro para Entornos Corporativos)
echo ========================================================
echo.
cd /d "%~dp0"

set PYTHON_CMD=
if exist "C:\Users\fabian.fierro\.venv_qs\Scripts\python.exe" (
    set PYTHON_CMD="C:\Users\fabian.fierro\.venv_qs\Scripts\python.exe"
) else (
    set PYTHON_CMD=python
)

echo Usando ejecutor: %PYTHON_CMD%
echo Iniciando Streamlit en http://127.0.0.1:8501 sin telemetria...
echo.

%PYTHON_CMD% -m streamlit run app.py --server.address=127.0.0.1 --server.port=8501 --browser.gatherUsageStats=false

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ========================================================
    echo AVISO: Se detecto un error al iniciar. Intentando con Python global...
    echo ========================================================
    python -m streamlit run app.py --server.address=127.0.0.1 --server.port=8501 --browser.gatherUsageStats=false
)

pause
