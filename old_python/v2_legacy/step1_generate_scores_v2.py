import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.calibration import IsotonicRegression

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
    base_dir = "results/v2_20k/02_Safety_Evaluation"
    data_dir = "data/v2_20k"
    models_dir = "models/v2_20k"
    
    os.makedirs(base_dir, exist_ok=True)
    sys.stdout = DualLogger(os.path.join(base_dir, "execution_log.txt"))
    
    print("="*60)
    print("v2_20k 評估與校正 Pipeline - 步驟一: 重度運算與校正推論")
    print("="*60)
    
    test1_path = os.path.join(data_dir, "test1_2000.pkl")
    test2_path = os.path.join(data_dir, "test2_2000.pkl")
    
    if not (os.path.exists(test1_path) and os.path.exists(test2_path)):
        print("錯誤: 找不到 test1 或 test2 檔案，請先執行 prepare_v2_20k_data.py")
        return
        
    df_test1 = pd.read_pickle(test1_path)
    df_test2 = pd.read_pickle(test2_path)
    
    X_3d_test1 = np.array(df_test1['hidden_state'].tolist())
    X_3d_test2 = np.array(df_test2['hidden_state'].tolist())
    
    models_list = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
    targets_list = ['y1', 'y2', 'y3']
    modes = ["baseline", "split"]
    
    for mode in modes:
        print(f"\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n執行校正模式: {mode.upper()}\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        
        metrics_records = []
        predictions_cache = {t: {} for t in targets_list}
        
        cache_dir = os.path.join(base_dir, "cache", mode)
        models_calib_dir = f"models/v2_20k/calibrated_isotonic/{mode}"
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(models_calib_dir, exist_ok=True)
        
        for target_name in targets_list:
            print(f"\n----------------------------------------\n處理目標任務: {target_name.upper()}\n----------------------------------------")
            
            y1_test1, y2_test1, y3_test1 = df_test1['y1'].values, df_test1['y2'].values, df_test1['y3'].values
            y1_test2, y2_test2, y3_test2 = df_test2['y1'].values, df_test2['y2'].values, df_test2['y3'].values
            
            for layer_num in range(1, 7):
                layer_calib_dir = os.path.join(models_calib_dir, f"layer_{layer_num}")
                os.makedirs(layer_calib_dir, exist_ok=True)
                
                X_test1 = X_3d_test1[:, layer_num - 1, :]
                X_test2 = X_3d_test2[:, layer_num - 1, :]
                
                predictions_cache[target_name][layer_num] = {
                    'bin_edges': {},
                    'splits': {
                        'test1': {},
                        'test2': {}
                    }
                }
                
                bin_edges_dict = {m: np.linspace(0.0, 1.0, 11) for m in models_list}
                
                for model_name in models_list:
                    model_path = os.path.join(models_dir, f"layer_{layer_num}", f"{model_name.lower()}_{target_name}_best.pkl")
                    if not os.path.exists(model_path):
                        continue
                        
                    clf = joblib.load(model_path)
                    
                    p_test1 = clf.predict_proba(X_test1)[:, 1]
                    p_test2 = clf.predict_proba(X_test2)[:, 1]
                    
                    if target_name in ['y1', 'y2']:
                        pre_cal_test1 = np.where(y1_test1 == 1, p_test1, 1.0 - p_test1)
                        pre_cal_test2 = np.where(y1_test2 == 1, p_test2, 1.0 - p_test2)
                    else:  # y3
                        pre_cal_test1 = p_test1
                        pre_cal_test2 = p_test2
                        
                    calib_save_path = f"{layer_calib_dir}/{model_name.lower()}_{target_name}_iso.pkl"
                    
                    if mode == "baseline":
                        iso = IsotonicRegression(out_of_bounds='clip')
                        iso.fit(pre_cal_test1, y3_test1)
                        
                        p_cal_test1 = iso.predict(pre_cal_test1)
                        p_cal_test2 = iso.predict(pre_cal_test2)
                        joblib.dump({'iso': iso}, calib_save_path)
                    else:  # split mode
                        mask_0 = (y1_test1 == 0)
                        mask_1 = (y1_test1 == 1)
                        iso_0 = IsotonicRegression(out_of_bounds='clip')
                        iso_1 = IsotonicRegression(out_of_bounds='clip')
                        
                        if np.sum(mask_0) > 0: iso_0.fit(pre_cal_test1[mask_0], y3_test1[mask_0])
                        else: iso_0.fit([0.0, 1.0], [0.0, 1.0])
                        
                        if np.sum(mask_1) > 0: iso_1.fit(pre_cal_test1[mask_1], y3_test1[mask_1])
                        else: iso_1.fit([0.0, 1.0], [0.0, 1.0])
                        
                        joblib.dump({'iso_0': iso_0, 'iso_1': iso_1}, calib_save_path)
                        
                        def predict_split(score_pre, y1_labels):
                            p_post = np.zeros_like(score_pre, dtype=float)
                            m0 = (y1_labels == 0)
                            m1 = (y1_labels == 1)
                            if np.sum(m0) > 0: p_post[m0] = iso_0.predict(score_pre[m0])
                            if np.sum(m1) > 0: p_post[m1] = iso_1.predict(score_pre[m1])
                            return p_post
                            
                        p_cal_test1 = predict_split(pre_cal_test1, y1_test1)
                        p_cal_test2 = predict_split(pre_cal_test2, y1_test2)
                        
                    splits_info = {
                        'test1': {'y_true': y3_test1, 'y_prob': p_cal_test1, 'y_prob_pre': p_test1, 'score_pre': pre_cal_test1, 'y1': y1_test1, 'y2': y2_test1, 'y3': y3_test1},
                        'test2': {'y_true': y3_test2, 'y_prob': p_cal_test2, 'y_prob_pre': p_test2, 'score_pre': pre_cal_test2, 'y1': y1_test2, 'y2': y2_test2, 'y3': y3_test2}
                    }
                    
                    predictions_cache[target_name][layer_num]['bin_edges'][model_name] = bin_edges_dict[model_name]
                    for split_name, s_data in splits_info.items():
                        predictions_cache[target_name][layer_num]['splits'][split_name][model_name] = s_data
                        
                        m = utils_calibration.calculate_all_metrics(s_data['y_true'], s_data['y_prob'])
                        ece = utils_calibration.calculate_ece(s_data['y_true'], s_data['y_prob'])
                        metrics_records.append({
                            'task': target_name,
                            'layer': layer_num,
                            'eval_set': split_name,
                            'model': model_name,
                            'brier': m['brier'],
                            'logloss': m['logloss'],
                            'reliability': m['reliability'],
                            'resolution': m['resolution'],
                            'uncertainty': m['uncertainty'],
                            'ece': ece
                        })
                        print(f"      [{mode.upper()} | {split_name.upper()} | Layer {layer_num} | {target_name.upper()} | {model_name}] Brier={m['brier']:.4f} | Rel={m['reliability']:.4f} | Res={m['resolution']:.4f} | Unc={m['uncertainty']:.4f} | ECE={ece:.4f} | LogLoss={m['logloss']:.4f}")
                        
        print(f"\n[儲存 {mode.upper()} 運算結果至快取...] ")
        pd.DataFrame(metrics_records).to_csv(os.path.join(cache_dir, "all_metrics_records.csv"), index=False)
        joblib.dump(predictions_cache, os.path.join(cache_dir, "calibrated_predictions.pkl"))
        
    print("\nStep 1 運算與快取寫入完畢！")

if __name__ == '__main__':
    main()
