import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.calibration import IsotonicRegression

# Add current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from unified_train import DataPreprocessor
import utils_calibration

import argparse

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    base_dir = "cache/v1_baseline/calibration"
    cache_dir = base_dir
    models_calib_dir = "cache/v1_baseline/calibration/models/calibrated_isotonic"
    
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(models_calib_dir, exist_ok=True)
    
    print("="*80)
    print(" [步驟 2] 開始進行雙軌機率校正訓練與預測值計算")
    print("="*80)
    
    try:
        # Load pre-split test sets
        test1_dict = joblib.load("data/test1.pkl")
        test2_dict = joblib.load("data/test2.pkl")
        
        # Load independent Eval Dataset
        prep_eval = DataPreprocessor("data/experiment_results_eval.pkl")
        prep_eval.load_data()
        X_3d_eval = prep_eval.extract_features()
        y_targets_eval = prep_eval.create_targets()
        
        y1_eval = y_targets_eval[0].values if hasattr(y_targets_eval[0], 'values') else y_targets_eval[0]
        y2_eval = y_targets_eval[1].values if hasattr(y_targets_eval[1], 'values') else y_targets_eval[1]
        y3_eval = y_targets_eval[2].values if hasattr(y_targets_eval[2], 'values') else y_targets_eval[2]
    except Exception as e:
        print(f"錯誤: 無法讀取資料集: {e}")
        return
        
    models_list = ['MLP', 'LGB', 'LR']
    targets_list = ['y2']  # Optimize: only run y2 as per request
    
    metrics_records = []
    predictions_cache = {t: {} for t in targets_list}
    
    for target_name in targets_list:
        print(f"\n----------------------------------------\n處理目標任務: {target_name.upper()}\n----------------------------------------")
        
        df_test1 = test1_dict[target_name]
        df_test2 = test2_dict[target_name]
        
        # Ground truths
        y1_test1, y2_test1, y3_test1 = df_test1['y1'].values, df_test1['y2'].values, df_test1['y3'].values
        y1_test2, y2_test2, y3_test2 = df_test2['y1'].values, df_test2['y2'].values, df_test2['y3'].values
        
        predictions_cache[target_name] = {}
        
        for layer_num in range(1, 7):
            layer_calib_dir = os.path.join(models_calib_dir, f"layer_{layer_num}")
            os.makedirs(layer_calib_dir, exist_ok=True)
            
            # Extract features for current layer
            X_test1 = np.array(df_test1['hidden_state'].tolist())[:, layer_num - 1, :]
            X_test2 = np.array(df_test2['hidden_state'].tolist())[:, layer_num - 1, :]
            X_eval = X_3d_eval[:, layer_num - 1, :]
            
            predictions_cache[target_name][layer_num] = {
                'splits': {
                    'test1': {},
                    'test2': {},
                    'eval': {}
                }
            }
            
            for model_name in models_list:
                candidate_paths = [
                    f"models/v1_baseline/unified_training/lgb_y2_all_models_y2_78k/layer_{layer_num}/{model_name.lower()}_{target_name}_best.pkl",
                    f"models/v1_baseline/unified_training/layer_{layer_num}/{model_name.lower()}_{target_name}_best.pkl",
                    f"models/v1_baseline/unified_training/lgb_y2_78k_ultimate/layer_{layer_num}/{model_name.lower()}_{target_name}_best.pkl",
                    f"models/v1_baseline/unified_training/lgb_y2/layer_{layer_num}/{model_name.lower()}_{target_name}_best.pkl"
                ]
                model_path = None
                for p in candidate_paths:
                    if os.path.exists(p):
                        model_path = p
                        break
                if model_path is None:
                    continue
                    
                clf = joblib.load(model_path)
                
                # Predict raw probabilities
                p_test1 = clf.predict_proba(X_test1)[:, 1]
                p_test2 = clf.predict_proba(X_test2)[:, 1]
                p_eval = clf.predict_proba(X_eval)[:, 1]
                
                # Calculate pre-calibration scores based on target task
                if target_name in ['y1', 'y2']:
                    pre_cal_test1 = np.where(y1_test1 == 1, p_test1, 1.0 - p_test1)
                    pre_cal_test2 = np.where(y1_test2 == 1, p_test2, 1.0 - p_test2)
                    pre_cal_eval  = np.where(y1_eval == 1,  p_eval,  1.0 - p_eval)
                else:  # y3
                    pre_cal_test1 = p_test1
                    pre_cal_test2 = p_test2
                    pre_cal_eval  = p_eval
                
                # 統一的 Isotonic Regression 校正 (不再區分 y1)
                iso_model = IsotonicRegression(out_of_bounds='clip')
                iso_model.fit(pre_cal_test1, y3_test1)
                
                # Save calibration pair
                calib_save_path = f"{layer_calib_dir}/{model_name.lower()}_{target_name}_iso.pkl"
                joblib.dump({'iso_model': iso_model}, calib_save_path)
                
                # Generate calibrated probabilities
                p_cal_test1 = iso_model.predict(pre_cal_test1)
                p_cal_test2 = iso_model.predict(pre_cal_test2)
                p_cal_eval = iso_model.predict(pre_cal_eval)
                
                # Prepare cache structure
                splits_info = {
                    'test1': {'y_true': y3_test1, 'y_prob': p_cal_test1, 'y_prob_pre': p_test1, 'score_pre': pre_cal_test1, 'y1': y1_test1, 'y2': y2_test1, 'y3': y3_test1},
                    'test2': {'y_true': y3_test2, 'y_prob': p_cal_test2, 'y_prob_pre': p_test2, 'score_pre': pre_cal_test2, 'y1': y1_test2, 'y2': y2_test2, 'y3': y3_test2},
                    'eval': {'y_true': y3_eval, 'y_prob': p_cal_eval, 'y_prob_pre': p_eval, 'score_pre': pre_cal_eval, 'y1': y1_eval, 'y2': y2_eval, 'y3': y3_eval}
                }
                
                for split_name, s_data in splits_info.items():
                    # Record for cache
                    predictions_cache[target_name][layer_num]['splits'][split_name][model_name] = s_data
                    
                    # Calculate and record metrics for diagnostic reporting
                    # Note: We calculate metrics on the full set, but plot_curves will split them for split_y plots!
                    m_raw = utils_calibration.calculate_all_metrics(s_data['y_true'], s_data['score_pre'])
                    m_cal = utils_calibration.calculate_all_metrics(s_data['y_true'], s_data['y_prob'])
                    
                    # Append records for logging
                    metrics_records.append({
                        'task': target_name,
                        'layer': layer_num,
                        'eval_set': split_name,
                        'model': model_name,
                        'raw_brier': m_raw['brier'],
                        'cal_brier': m_cal['brier'],
                        'raw_logloss': m_raw['logloss'],
                        'cal_logloss': m_cal['logloss']
                    })
                    
                    print(f"    [{model_name} | 層: {layer_num} | 評估集: {split_name}]")
                    print(f"      Raw Brier: {m_raw['brier']:.5f} -> Cal Brier: {m_cal['brier']:.5f}")
                    print(f"      Raw LogLoss: {m_raw['logloss']:.5f} -> Cal LogLoss: {m_cal['logloss']:.5f}")
                    
    # Save the cache files
    print("\n[OK] 正在寫入預測值快取與指標日誌...")
    df_metrics = pd.DataFrame(metrics_records)
    df_metrics.to_csv(os.path.join(cache_dir, "all_metrics_records.csv"), index=False)
    joblib.dump(predictions_cache, os.path.join(cache_dir, "calibrated_predictions.pkl"))
    print("雙軌校正與計算完成！快取儲存於 results/v1_baseline/safety_guardrails_evaluation/cache/calibrated_predictions.pkl")

if __name__ == '__main__':
    main()
