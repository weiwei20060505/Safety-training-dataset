import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.calibration import IsotonicRegression

# Ensure evaluation_pipeline can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation_pipeline.step3_plot_histograms import plot_quadrant_histograms
from unified_train import DataPreprocessor
import utils_calibration

utils_calibration.setup_chinese_font()

def test_all_histograms():
    print("Testing histogram generation on simplified 10k Test split for y1, y2, y3 (Option B: No downsampling)...")
    
    # 1. Load initial 10,000 training dataset
    train_path = "data/experiment_results_train_10000.pkl"
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found.")
        return
        
    preprocessor = DataPreprocessor(train_path)
    df_10000 = preprocessor.load_data()
    X_3d_train = preprocessor.extract_features()
    y_targets_train = preprocessor.create_targets()
    
    # Pre-calculate target labels for df_10000
    df_10000['y1'] = y_targets_train[0]
    df_10000['y2'] = y_targets_train[1]
    df_10000['y3'] = y_targets_train[2]
    
    layer_num = 6
    
    for target_idx, target_name in enumerate(['y1', 'y2', 'y3']):
        print(f"\n--- Processing Target: {target_name.upper()} ---")
        y_train = y_targets_train[target_idx]
        
        # 2. Re-create the 20% Test Set split (matching the training split stratify by current target)
        train_val_idx, test_idx = train_test_split(
            df_10000.index, test_size=0.2, random_state=42, stratify=y_train
        )
        # Split test_idx in half to get test1 (Calibration) and test2 (Validation)
        test1_idx, test2_idx = train_test_split(
            test_idx, test_size=0.5, random_state=42
        )
        
        df_test1 = df_10000.loc[test1_idx].copy()
        
        # Extract features for layer
        X_test1 = X_3d_train[test1_idx, layer_num - 1, :]
        
        y1_test1 = df_test1['y1'].values
        y2_test1 = df_test1['y2'].values
        y3_test1 = df_test1['y3'].values
        
        # 3. Load SGD model
        model_path = f"models/unified_training/layer_{layer_num}/sgd_{target_name}_best.pkl"
        if not os.path.exists(model_path):
            print(f"Error: {model_path} not found.")
            continue
            
        clf = joblib.load(model_path)
        
        # 4. Predict raw probs
        p_test1 = clf.predict_proba(X_test1)[:, 1]
        
        # 依據任務目標嚴密計算 pre_cal 分數
        if target_name in ['y1', 'y2']:
            pre_cal_test1 = np.where(y1_test1 == 1, p_test1, 1.0 - p_test1)
        else:  # y3 任務：絕對不可反轉！直接使用原始預測機率！
            pre_cal_test1 = p_test1
        
        # 5. Fit Isotonic Regression (Option B: No downsampling)
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(pre_cal_test1, y3_test1)
        
        # 6. Predict calibrated probs
        p_cal_test1 = iso.predict(pre_cal_test1)
        
        # 7. Plot
        save_path = f"results/safety_guardrails_evaluation/test_quick_histogram_{target_name}.png"
        plot_quadrant_histograms(
            pre_cal_test1, p_cal_test1, y1_test1, y2_test1, y3_test1,
            layer_num, "SGD", "10k Test Split (Option B)", "TEST1_CALIB", target_name, save_path
        )
        print(f"Success! Test histogram saved to {save_path}")

if __name__ == "__main__":
    test_all_histograms()
