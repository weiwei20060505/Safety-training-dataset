"""
分析 test1/test2 樣本比例以及模型在 y1=0/y1=1 分流下，修正前後預測分數在 10 個 Bin 區間的分布日誌 (全 6 層特徵版)。
"""

import os
import sys
import io
import numpy as np
import pandas as pd
import joblib

# 處理 Windows 控制台 cp950 Unicode 編碼相容性
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def analyze_test_splits(test1_path, test2_path):
    log_dir = "results"
    os.makedirs(log_dir, exist_ok=True)
    split_log_path = os.path.join(log_dir, "check_test_splits.log")
    
    with open(split_log_path, 'w', encoding='utf-8') as lf:
        header = "=" * 80 + "\n [部分一] test1 與 test2 資料集樣本規模與標籤占比分析\n" + "=" * 80 + "\n"
        print(header, end="")
        lf.write(header)
        
        for split_name, path in [('test1', test1_path), ('test2', test2_path)]:
            if not os.path.exists(path):
                msg = f"⚠️ 找不到 {split_name} 的資料檔: {path}\n"
                print(msg, end="")
                lf.write(msg)
                continue
                
            data = joblib.load(path)
            msg = f"\n📌 【{split_name.upper()}】資料集資訊:\n"
            print(msg, end="")
            lf.write(msg)
            
            for task_name in ['y1', 'y2', 'y3']:
                df_task = data[task_name]
                total_samples = len(df_task)
                
                y_vals = df_task[task_name].values
                pos_count = np.sum(y_vals == 1)
                neg_count = np.sum(y_vals == 0)
                pos_pct = pos_count / total_samples * 100.0 if total_samples > 0 else 0
                neg_pct = neg_count / total_samples * 100.0 if total_samples > 0 else 0
                
                msg_task = (f"  └─ 任務 {task_name.upper()} (總筆數: {total_samples}):\n"
                            f"     * 正樣本 (1): {pos_count:5d} 筆 ({pos_pct:6.2f}%)\n"
                            f"     * 負樣本 (0): {neg_count:5d} 筆 ({neg_pct:6.2f}%)\n")
                print(msg_task, end="")
                lf.write(msg_task)
                
    print(f"\n✨ 資料集分割比例日誌已成功寫入: {split_log_path}")

def analyze_score_proportions_all_layers(cache_path, log_path, target='y1', layers=[1, 2, 3, 4, 5, 6], models=['LGB', 'SGD', 'MLP', 'LR', 'RF']):
    if not os.path.exists(cache_path):
        print(f"❌ 找不到預測值快取: {cache_path}")
        return
        
    predictions_cache = joblib.load(cache_path)
    edges = np.linspace(0.0, 1.0, 11)
    bin_labels = [f'{i/10:.1f}-{(i+1)/10:.1f}' for i in range(10)]
    
    with open(log_path, 'w', encoding='utf-8') as f:
        # 寫入開頭標題
        f.write("=" * 80 + "\n")
        f.write(f" 原始分數 (Raw Score) vs 校正分數 (Calibrated Score) 10-Bin 比例分布對比日誌\n")
        f.write(f" 任務: {target.upper()} | 包含層數: Layer 1-6 | 評估模型: {models}\n")
        f.write("=" * 80 + "\n\n")
        
        for layer in layers:
            f.write(f"########################################################################\n")
            f.write(f"                   ✨ 特徵層: 【LAYER {layer}】\n")
            f.write(f"########################################################################\n\n")
            
            layer_data = predictions_cache[target][layer]['splits']
            
            for split in ['test1', 'test2']:
                f.write(f"  ====================================================================\n")
                f.write(f"    評估資料集: {split.upper()} | Layer {layer}\n")
                f.write(f"  ====================================================================\n\n")
                
                for model_name in models:
                    if model_name not in layer_data[split]:
                        f.write(f"  ⚠️ 找不到模型 {model_name} 在 {split} (Layer {layer}) 的預測快取資料。\n\n")
                        continue
                        
                    data = layer_data[split][model_name]
                    score_pre = np.array(data['score_pre'])
                    y_prob_cal = np.array(data['y_prob'])
                    y1_labels = np.array(data['y1'])
                    
                    for group_val, group_name in [(0, 'Safe (y1=0)'), (1, 'Unsafe (y1=1)')]:
                        mask = (y1_labels == group_val)
                        if np.sum(mask) == 0:
                            continue
                            
                        pre_g = score_pre[mask]
                        post_g = y_prob_cal[mask]
                        n_samples = len(pre_g)
                        
                        f.write(f"    🔹 模型: {model_name} | 分流組別: {group_name} | 總樣本數: {n_samples}\n")
                        f.write(f"    --------------------------------------------------------------------\n")
                        f.write(f"    | Bin 區間    | 修正前數量 | 修正前比例 | 修正後數量 | 修正後比例 |\n")
                        f.write(f"    | :---        | :---:      | :---:      | :---:      | :---:      |\n")
                        
                        # 計算各 Bin 數量與比例
                        bin_ids_pre = np.digitize(pre_g, edges)
                        bin_ids_post = np.digitize(post_g, edges)
                        
                        for b in range(1, 11):
                            mask_pre = (bin_ids_pre == b)
                            mask_post = (bin_ids_post == b)
                            if b == 10:
                                mask_pre = mask_pre | (pre_g == 1.0)
                                mask_post = mask_post | (post_g == 1.0)
                                
                            cnt_pre = np.sum(mask_pre)
                            cnt_post = np.sum(mask_post)
                            
                            pct_pre = cnt_pre / n_samples * 100.0 if n_samples > 0 else 0.0
                            pct_post = cnt_post / n_samples * 100.0 if n_samples > 0 else 0.0
                            
                            f.write(f"    | {bin_labels[b-1]:9s} | {cnt_pre:10d} | {pct_pre:9.2f}% | {cnt_post:10d} | {pct_post:9.2f}% |\n")
                            
                        f.write("\n")
        
    print(f"✨ 全 6 層比例對比 Log 已成功寫入: {log_path}")

def main():
    test1_path = "data/test1.pkl"
    test2_path = "data/test2.pkl"
    cache_path = "results/safety_guardrails_evaluation/cache/calibrated_predictions.pkl"
    log_path = "results/score_proportion_log.txt"
    
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # 執行第一部分分析
    analyze_test_splits(test1_path, test2_path)
    
    # 執行第二部分分析並輸出 Log (遍歷全 6 層)
    analyze_score_proportions_all_layers(cache_path, log_path, target='y1', layers=[1, 2, 3, 4, 5, 6], models=['LGB', 'SGD', 'MLP', 'LR', 'RF'])

if __name__ == "__main__":
    main()
