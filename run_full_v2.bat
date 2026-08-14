@echo off
echo Running V2 Full Pipeline...
echo ============================

echo 1. Train models
call .venv\Scripts\python.exe pipeline_v2_framework\train_framework_models.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo 2. Calibrate models
call .venv\Scripts\python.exe pipeline_v2_framework\calibrate_framework_models.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo 3. Plot stage 2
call .venv\Scripts\python.exe pipeline_v2_framework\plot_framework_stage2_v2.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo ============================
echo All Done!
