@echo off
echo Running V1 Full 5-Model Ultimate Evaluation (78k samples, Y2 task, SGD, MLP, LGB, LR, RF)...

.venv\Scripts\python.exe pipeline_v1_baseline\unified_train.py --no_pca --train_data data\v1_train_full.pkl --val_data data\v1_val.pkl --output_suffix all_models_y2_78k
if %errorlevel% neq 0 exit /b %errorlevel%

echo Done! All 5 models evaluated successfully.
