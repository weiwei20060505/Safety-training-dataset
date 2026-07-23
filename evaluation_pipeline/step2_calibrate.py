import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.calibration import IsotonicRegression

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unified_train import DataPreprocessor
import utils_calibration

def main():
    base_dir = "results/safety_guardrails_evaluation"
    cache_dir = os.path.join(base_dir, "cache")
    models_calib_dir = "models/calibrated_isotonic"
    
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
        
    models_list = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
    targets_list = ['y1', 'y2', 'y3']
    
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
                model_path = f"results/unified_training/layer_{layer_num}/{model_name.lower()}_{target_name}_best.pkl"
                if not os.path.exists(model_path):
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
                
                # 依據 y1 標籤分流進行 Isotonic Regression 擬合
                mask_0_test1 = (y1_test1 == 0)
                mask_1_test1 = (y1_test1 == 1)
                
                iso_0 = IsotonicRegression(out_of_bounds='clip')
                iso_1 = IsotonicRegression(out_of_bounds='clip')
                
                # Fit separate calibrators without undersampling
                if np.sum(mask_0_test1) > 0:
                    iso_0.fit(pre_cal_test1[mask_0_test1], y3_test1[mask_0_test1])
                else:
                    iso_0.fit([0.0, 1.0], [0.0, 1.0])
                    
                if np.sum(mask_1_test1) > 0:
                    iso_1.fit(pre_cal_test1[mask_1_test1], y3_test1[mask_1_test1])
                else:
                    iso_1.fit([0.0, 1.0], [0.0, 1.0])
                
                # Save calibration pair
                calib_save_path = f"{layer_calib_dir}/{model_name.lower()}_{target_name}_iso.pkl"
                joblib.dump({'iso_0': iso_0, 'iso_1': iso_1}, calib_save_path)
                
                # Split prediction helper
                def predict_split(score_pre, y1_labels):
                    prob_post = np.zeros_like(score_pre, dtype=float)
                    m0 = (y1_labels == 0)
                    m1 = (y1_labels == 1)
                    if np.sum(m0) > 0:
                        prob_post[m0] = iso_0.predict(score_pre[m0])
                    if np.sum(m1) > 0:
                        prob_post[m1] = iso_1.predict(score_pre[m1])
                    return prob_post
                
                # Generate calibrated probabilities
                p_cal_test1 = predict_split(pre_cal_test1, y1_test1)
                p_cal_test2 = predict_split(pre_cal_test2, y1_test2)
                p_cal_eval = predict_split(pre_cal_eval, y1_eval)
                
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
    print("雙軌校正與計算完成！快取儲存於 results/safety_guardrails_evaluation/cache/calibrated_predictions.pkl")

if __name__ == '__main__':
    main()
