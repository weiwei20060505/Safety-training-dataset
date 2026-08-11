@echo off
echo Running V2 Full Pipeline...
echo ============================

echo 1. Train models (with_pca)
call .venv\Scripts\python.exe pipeline_v2_framework\train_framework_models.py --use_pca
if %errorlevel% neq 0 exit /b %errorlevel%

echo 2. Train models (without_pca)
call .venv\Scripts\python.exe pipeline_v2_framework\train_framework_models.py --no_pca
if %errorlevel% neq 0 exit /b %errorlevel%

echo 3. Calibrate models (with_pca)
call .venv\Scripts\python.exe pipeline_v2_framework\calibrate_framework_models.py --use_pca
if %errorlevel% neq 0 exit /b %errorlevel%

echo 4. Calibrate models (without_pca)
call .venv\Scripts\python.exe pipeline_v2_framework\calibrate_framework_models.py --no_pca
if %errorlevel% neq 0 exit /b %errorlevel%

echo 5. Plot stage 2 (with_pca)
call .venv\Scripts\python.exe pipeline_v2_framework\plot_framework_stage2_v2.py --use_pca
if %errorlevel% neq 0 exit /b %errorlevel%

echo 6. Plot stage 2 (without_pca)
call .venv\Scripts\python.exe pipeline_v2_framework\plot_framework_stage2_v2.py --no_pca
if %errorlevel% neq 0 exit /b %errorlevel%

echo ============================
echo All Done!
