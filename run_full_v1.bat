@echo off
echo ===================================================
echo  Running V1 Baseline Pipeline (Full Evaluation)
echo ===================================================

echo [1/4] Running Unified Training (MLP, LGB, LR on Y2)...
.venv\Scripts\python.exe pipeline_v1_baseline\unified_train.py --train_data data\v1_train_full.pkl --val_data data\v1_val.pkl --output_suffix all_models_y2_78k
if %errorlevel% neq 0 exit /b %errorlevel%

echo [2/4] Running Probability Calibration (Isotonic Regression)...
.venv\Scripts\python.exe pipeline_v1_baseline\step2_calibrate.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo [3/4] Generating Custom Diagnostic Plots (Trends, Reliability, Joint Calibration)...
.venv\Scripts\python.exe pipeline_v1_baseline\plot_v1_custom.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo [4/4] Generating Standard Combined Plots...
.venv\Scripts\python.exe pipeline_v1_baseline\step4_combine_plots.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo ===================================================
echo  V1 Baseline Pipeline Completed Successfully!
echo ===================================================
