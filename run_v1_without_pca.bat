@echo off
echo Running V1 without_pca pipeline...

echo [1/4] Running unified_train.py without PCA...
.venv\Scripts\python.exe pipeline_v1_baseline\unified_train.py --no_pca
if %errorlevel% neq 0 exit /b %errorlevel%

echo [2/4] Running step2_calibrate.py without PCA...
.venv\Scripts\python.exe pipeline_v1_baseline\step2_calibrate.py --no_pca
if %errorlevel% neq 0 exit /b %errorlevel%

echo [3/4] Running step3_plot.py without PCA...
.venv\Scripts\python.exe pipeline_v1_baseline\step3_plot.py --no_pca
if %errorlevel% neq 0 exit /b %errorlevel%

echo [4/4] Running step4_combine_plots.py without PCA...
.venv\Scripts\python.exe pipeline_v1_baseline\step4_combine_plots.py --no_pca
if %errorlevel% neq 0 exit /b %errorlevel%

echo Done!
