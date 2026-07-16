import os
import sys
import argparse
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
    parser = argparse.ArgumentParser(description="LLM Safety Probe Pipeline - Step 1: Score Generation & Calibration")
    parser.add_argument("--mode", type=str, choices=["baseline", "split", "all"], default="all",
                        help="Calibration mode: baseline (unified), split (conditional y1-split), or all (run both)")
    args = parser.parse_args()
    
    base_dir = "results/safety_guardrails_evaluation"
    os.makedirs(base_dir, exist_ok=True)
    
    # Set up dual logging to terminal and execution_log.txt
    sys.stdout = DualLogger(os.path.join(base_dir, "execution_log.txt"))
    
    print("="*60)
    print(f"LLM Safety Probe Pipeline - 步驟一: 重度運算 (模式: {args.mode.upper()})")
    print("="*60)
    
    print("[1] 載入原始資料集...")
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
    
    # Determine which modes to run
    modes_to_run = ["baseline", "split"] if args.mode == "all" else [args.mode]
    
    for mode in modes_to_run:
        print(f"\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n執行校正模式: {mode.upper()}\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        
        # 💡 [陷阱一解決策略] 每次進入新模式，務必清空與重置快取變數！
        metrics_records = []
        predictions_cache = {
            'data_aug': {t: {} for t in targets_list},
            'data_align': {t: {} for t in targets_list}
        }
        
        cache_dir = os.path.join(base_dir, "cache", mode)
        models_calib_dir = f"models/calibrated_isotonic/{mode}"
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(models_calib_dir, exist_ok=True)
        
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
                    # Ensure layer subdirectory exists for saving calibrators
                    layer_calib_dir = os.path.join(models_calib_dir, f"layer_{layer_num}")
                    os.makedirs(layer_calib_dir, exist_ok=True)
                    
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
                        
                        # 依據任務目標嚴密計算 pre_cal 分數
                        if target_name in ['y1', 'y2']:
                            pre_cal_test1 = np.where(y1_test1 == 1, p_test1, 1.0 - p_test1)
                            pre_cal_test2 = np.where(y1_test2 == 1, p_test2, 1.0 - p_test2)
                            pre_cal_cross = np.where(y1_cross == 1, p_cross, 1.0 - p_cross)
                            pre_cal_eval  = np.where(y1_eval == 1,  p_eval,  1.0 - p_eval)
                        else:  # y3 任務：絕對不可反轉！直接使用原始預測機率！
                            pre_cal_test1 = p_test1
                            pre_cal_test2 = p_test2
                            pre_cal_cross = p_cross
                            pre_cal_eval  = p_eval
                            
                        # Save model file path
                        calib_save_path = f"{layer_calib_dir}/{model_name.lower()}_{target_name}_{dataset_key}_iso.pkl"
                        
                        # Calibration logic based on mode
                        if mode == "baseline":
                            # Fit single unified Isotonic Regression
                            iso = IsotonicRegression(out_of_bounds='clip')
                            iso.fit(pre_cal_test1, y3_test1)
                            
                            p_cal_test1 = iso.predict(pre_cal_test1)
                            p_cal_test2 = iso.predict(pre_cal_test2)
                            p_cal_cross = iso.predict(pre_cal_cross)
                            p_cal_eval = iso.predict(pre_cal_eval)
                            
                            # 💡 儲存單一校正模型
                            joblib.dump({'iso': iso}, calib_save_path)
                            
                        else:  # split mode
                            # 1. 依據模型自己的原始機率分流，絕不使用真實標籤 y1 避免洩漏！
                            y1_pred_test1 = (p_test1 >= 0.5).astype(int)
                            mask_0_test1 = (y1_pred_test1 == 0)
                            mask_1_test1 = (y1_pred_test1 == 1)
                            
                            iso_0 = IsotonicRegression(out_of_bounds='clip')
                            iso_1 = IsotonicRegression(out_of_bounds='clip')
                            
                            # 2. 分別訓練：特徵 X 是 pre_cal，目標 Y 必須是 y3_test1！絕對不可把 y1 餵給 Y！
                            if np.sum(mask_0_test1) > 10:
                                iso_0.fit(pre_cal_test1[mask_0_test1], y3_test1[mask_0_test1])
                            else:
                                iso_0.fit([0.0, 1.0], [0.0, 1.0])
                                
                            if np.sum(mask_1_test1) > 10:
                                iso_1.fit(pre_cal_test1[mask_1_test1], y3_test1[mask_1_test1])
                            else:
                                iso_1.fit([0.0, 1.0], [0.0, 1.0])
                                
                            # 💡 儲存雙軌條件校正模型對
                            joblib.dump({'iso_0': iso_0, 'iso_1': iso_1}, calib_save_path)
                            
                            # 3. 條件推論函數：未知資料集依據自己的預測分數分流
                            def predict_split_calibration(prob_raw, score_pre):
                                prob_post = np.zeros_like(score_pre, dtype=float)
                                m0 = (prob_raw < 0.5)
                                m1 = (prob_raw >= 0.5)
                                if np.sum(m0) > 0:
                                    prob_post[m0] = iso_0.predict(score_pre[m0])
                                if np.sum(m1) > 0:
                                    prob_post[m1] = iso_1.predict(score_pre[m1])
                                return prob_post
                                
                            p_cal_test1 = predict_split_calibration(p_test1, pre_cal_test1)
                            p_cal_test2 = predict_split_calibration(p_test2, pre_cal_test2)
                            p_cal_cross = predict_split_calibration(p_cross, pre_cal_cross)
                            p_cal_eval = predict_split_calibration(p_eval, pre_cal_eval)
                        
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
                            print(f"    [日誌 - {dataset_title} - 任務: {target_name} | 層數: {layer_num} | 評估集: {split_name} | 模型: {model_name} | 模式: {mode}]")
                            print(f"      Brier Score: {m['brier']:.5f} | Log Loss: {m['logloss']:.5f} | Reliability: {m['reliability']:.5f} | Resolution: {m['resolution']:.5f} | Uncertainty: {m['uncertainty']:.5f}")
        
        # Save the cache files for this mode
        print(f"\n[2] 儲存 {mode.upper()} 運算結果至快取目錄...")
        df_metrics = pd.DataFrame(metrics_records)
        df_metrics.to_csv(os.path.join(cache_dir, "all_metrics_records.csv"), index=False)
        joblib.dump(predictions_cache, os.path.join(cache_dir, "calibrated_predictions.pkl"))
        print(f"模式 {mode.upper()} 執行完畢！已成功將模型與快取寫入對應資料夾。")
        
    print("\n所有指定模式皆執行完畢！")

if __name__ == '__main__':
    main()
