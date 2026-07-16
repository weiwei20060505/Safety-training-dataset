import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.calibration import IsotonicRegression

# Add parent directory to sys.path to import core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified_train import DataPreprocessor
import utils_calibration

class DualLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")
    def write(self, message):
        try:
            self.terminal.write(message)
        except UnicodeEncodeError:
            encoding = getattr(self.terminal, 'encoding', 'utf-8') or 'utf-8'
            self.terminal.write(message.encode(encoding, errors='replace').decode(encoding))
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def main():
    base_dir = "results/safety_guardrails_evaluation"
    cache_dir = os.path.join(base_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Set up dual logging to terminal and execution_log.txt
    sys.stdout = DualLogger(os.path.join(base_dir, "execution_log.txt"))
    
    print("="*60)
    print("LLM Safety Probe Pipeline - 步驟一: 重度運算 (推論、拆分校正、存 Cache)")
    print("="*60)
    
    print("[1] 載入資料集...")
    try:
        # Load datasets from data/ directory
        aug_test1_dict = joblib.load("data/augmented_test1.pkl")
        aug_test2_dict = joblib.load("data/augmented_test2.pkl")
        align_test1_dict = joblib.load("data/aligned_test1.pkl")
        align_test2_dict = joblib.load("data/aligned_test2.pkl")
        
        # Load independent Eval Dataset
        prep_eval = DataPreprocessor("data/experiment_results_eval.pkl")
        prep_eval.load_data()
        X_3d_eval = prep_eval.extract_features()
        y_targets_eval = prep_eval.create_targets()
        
        y1_eval = y_targets_eval[0].values if hasattr(y_targets_eval[0], 'values') else y_targets_eval[0]
        y3_eval = y_targets_eval[2].values if hasattr(y_targets_eval[2], 'values') else y_targets_eval[2]
        
    except Exception as e:
        print(f"錯誤: 無法讀取資料集: {e}")
        return
        
    models_list = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
    targets_list = ['y1', 'y2', 'y3']
    
    datasets = [
        ('data_aug', 'augmented_test1.pkl', 'augmented_test2.pkl', 'aligned_test2.pkl', '資料增強'),
        ('data_align', 'aligned_test1.pkl', 'aligned_test2.pkl', 'augmented_test2.pkl', '資料對齊')
    ]
    
    # Metrics records to save as CSV
    metrics_records = []
    # Nested dictionary for predictions cache to save as PKL
    predictions_cache = {
        'data_aug': {t: {} for t in targets_list},
        'data_align': {t: {} for t in targets_list}
    }
    
    for dataset_key, test1_file, test2_file, cross_file, dataset_title in datasets:
        print(f"\n========================================\n處理資料組: {dataset_title} ({dataset_key})\n========================================\n")
        
        # Load group-specific datasets
        test1_dict = joblib.load(f"data/{test1_file}")
        test2_dict = joblib.load(f"data/{test2_file}")
        cross_dict = joblib.load(f"data/{cross_file}")
        cross_name = 'aligned_test2' if dataset_key == 'data_aug' else 'augmented_test2'
        
        for target_name in targets_list:
            print(f"\n----------------------------------------\n處理目標任務: {target_name.upper()}\n----------------------------------------")
            
            df_test1 = test1_dict[target_name]
            df_test2 = test2_dict[target_name]
            df_cross = cross_dict[target_name]
            
            for layer_num in range(1, 7):
                print(f"  - 處理第 {layer_num} 層...")
                
                # Extract hidden states
                X_test1 = np.array(df_test1['hidden_state'].tolist())[:, layer_num - 1, :]
                X_test2 = np.array(df_test2['hidden_state'].tolist())[:, layer_num - 1, :]
                X_cross = np.array(df_cross['hidden_state'].tolist())[:, layer_num - 1, :]
                X_eval = X_3d_eval[:, layer_num - 1, :]
                
                # Ground truths
                y1_test1, y3_test1 = df_test1['y1'].values, df_test1['y3'].values
                y1_test2, y3_test2 = df_test2['y1'].values, df_test2['y3'].values
                y1_cross, y3_cross = df_cross['y1'].values, df_cross['y3'].values
                
                predictions_cache[dataset_key][target_name][layer_num] = {
                    'bin_edges': {},
                    'splits': {
                        'test1': {},
                        'test2': {},
                        'test2_cross': {},
                        'eval': {}
                    }
                }
                
                bin_edges_dict = {m: np.linspace(0.0, 1.0, 11) for m in models_list}
                
                for model_name in models_list:
                    model_path = f"models/unified_training/layer_{layer_num}/{model_name.lower()}_{target_name}_best.pkl"
                    
                    if not os.path.exists(model_path):
                        continue
                        
                    clf = joblib.load(model_path)
                    
                    # Predict raw probabilities
                    p_test1 = clf.predict_proba(X_test1)[:, 1]
                    p_test2 = clf.predict_proba(X_test2)[:, 1]
                    p_cross = clf.predict_proba(X_cross)[:, 1]
                    p_eval = clf.predict_proba(X_eval)[:, 1]
                    
                    # Formulation based on target_name
                    if target_name in ['y1', 'y2']:
                        pre_cal_test1 = np.where(y1_test1 == 1, p_test1, 1.0 - p_test1)
                        pre_cal_test2 = np.where(y1_test2 == 1, p_test2, 1.0 - p_test2)
                        pre_cal_cross = np.where(y1_cross == 1, p_cross, 1.0 - p_cross)
                        pre_cal_eval = np.where(y1_eval == 1, p_eval, 1.0 - p_eval)
                    else:  # y3
                        pre_cal_test1 = np.where(y3_test1 == 1, p_test1, 1.0 - p_test1)
                        pre_cal_test2 = np.where(y3_test2 == 1, p_test2, 1.0 - p_test2)
                        pre_cal_cross = np.where(y3_cross == 1, p_cross, 1.0 - p_cross)
                        pre_cal_eval = np.where(y3_eval == 1, p_eval, 1.0 - p_eval)
                        
                    # Calibrate on test1 using Isotonic Regression
                    iso = IsotonicRegression(out_of_bounds='clip')
                    iso.fit(pre_cal_test1, y3_test1)
                    
                    p_cal_test1 = iso.predict(pre_cal_test1)
                    p_cal_test2 = iso.predict(pre_cal_test2)
                    p_cal_cross = iso.predict(pre_cal_cross)
                    p_cal_eval = iso.predict(pre_cal_eval)
                    
                    # Cache predictions for plotting later
                    splits_info = {
                        'test1': {'y_true': y3_test1, 'y_prob': p_cal_test1, 'y_prob_pre': p_test1, 'y1': y1_test1, 'y3': y3_test1},
                        'test2': {'y_true': y3_test2, 'y_prob': p_cal_test2, 'y_prob_pre': p_test2, 'y1': y1_test2, 'y3': y3_test2},
                        'test2_cross': {'y_true': y3_cross, 'y_prob': p_cal_cross, 'y_prob_pre': p_cross, 'y1': y1_cross, 'y3': y3_cross},
                        'eval': {'y_true': y3_eval, 'y_prob': p_cal_eval, 'y_prob_pre': p_eval, 'y1': y1_eval, 'y3': y3_eval}
                    }
                    
                    predictions_cache[dataset_key][target_name][layer_num]['bin_edges'][model_name] = bin_edges_dict[model_name]
                    for split_name, s_data in splits_info.items():
                        predictions_cache[dataset_key][target_name][layer_num]['splits'][split_name][model_name] = s_data
                        
                        # Calculate and record metrics
                        m = utils_calibration.calculate_all_metrics(s_data['y_true'], s_data['y_prob'])
                        metrics_records.append({
                            'data_group': 'Data Aug' if dataset_key == 'data_aug' else 'Data Align',
                            'task': target_name,
                            'layer': layer_num,
                            'eval_set': split_name if split_name != 'test2_cross' else cross_name,
                            'model': model_name,
                            'brier': m['brier'],
                            'logloss': m['logloss'],
                            'reliability': m['reliability'],
                            'resolution': m['resolution'],
                            'uncertainty': m['uncertainty']
                        })
                        
                        # Print detailed log
                        print(f"    [日誌 - {dataset_title} - 任務: {target_name} | 層數: {layer_num} | 評估集: {split_name} | 模型: {model_name}]")
                        print(f"      Brier Score: {m['brier']:.5f} | Log Loss: {m['logloss']:.5f} | Reliability: {m['reliability']:.5f} | Resolution: {m['resolution']:.5f} | Uncertainty: {m['uncertainty']:.5f}")
    
    # Save the cache files
    print("\n[2] 儲存運算結果至快取目錄...")
    df_metrics = pd.DataFrame(metrics_records)
    df_metrics.to_csv(os.path.join(cache_dir, "all_metrics_records.csv"), index=False)
    joblib.dump(predictions_cache, os.path.join(cache_dir, "calibrated_predictions.pkl"))
    
    print("步驟一執行完畢！已成功將數據存入 results/safety_guardrails_evaluation/cache/")

if __name__ == '__main__':
    main()
