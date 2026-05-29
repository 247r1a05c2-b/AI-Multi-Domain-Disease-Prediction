@echo off
echo ========================================
echo   BioVision AI — Startup Script
echo ========================================
echo.

REM Activate conda environment if it exists
call conda activate biovision 2>nul || (
    echo Creating conda environment...
    call conda create -n biovision python=3.10 -y
    call conda activate biovision
)

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Generating sample datasets...
python setup_datasets.py

echo.
echo Launching BioVision AI...
streamlit run app.py

pause
