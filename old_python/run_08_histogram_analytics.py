import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.calibration import IsotonicRegression

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

def plot_quadrant_histograms(pre_scores, post_scores, y_hat, y_gt, y3_gt, layer_num, model_name, dataset_title, split_name, target_name, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"第 {layer_num} 層隱藏狀態 - {model_name} 校正前後分數分佈 ({dataset_title} - {split_name} - 任務: {target_name.upper()})", fontsize=16, fontweight='bold')
    
    # Groups definition: (y_hat, y_gt)
    if target_name in ['y1', 'y2']:
        groups = [
            ((0, 0), "Group 1: Guardrail Safe (0) | GT Safe (0)", axes[0, 0]),
            ((0, 1), "Group 2: Guardrail Safe (0) | GT Unsafe (1)", axes[0, 1]),
            ((1, 0), "Group 3: Guardrail Unsafe (1) | GT Safe (0)", axes[1, 0]),
            ((1, 1), "Group 4: Guardrail Unsafe (1) | GT Unsafe (1)", axes[1, 1]),
        ]
    else: # y3
        groups = [
            ((0, 0), "Group 1: Pred Inconsistent (0) | GT Inconsistent (0)", axes[0, 0]),
            ((0, 1), "Group 2: Pred Inconsistent (0) | GT Consistent (1)", axes[0, 1]),
            ((1, 0), "Group 3: Pred Consistent (1) | GT Inconsistent (0)", axes[1, 0]),
            ((1, 1), "Group 4: Pred Consistent (1) | GT Consistent (1)", axes[1, 1]),
        ]
        
    # Print detailed quadrant statistics for text logging
    print(f"      [四象限直方圖數據 - 任務: {target_name.upper()} | 模型: {model_name} | 層數: {layer_num} | 評估集: {split_name}]")
    
    for (hat_val, gt_val), title, ax in groups:
        mask = (y_hat == hat_val) & (y_gt == gt_val)
        pre = pre_scores[mask]
        post = post_scores[mask]
        
        y_true_group = y3_gt[mask]
        n_samples = len(y_true_group)
        
        if n_samples > 0:
            brier, rel, res, unc = utils_calibration.brier_score_decomposition(y_true_group, post, n_bins=10)
            subtitle = f"Brier: {brier:.4f} | Rel(↓): {rel:.4f} | Res(↑): {res:.4f} | Unc: {unc:.4f}"
            mean_pre, std_pre = np.mean(pre), np.std(pre)
            mean_post, std_post = np.mean(post), np.std(post)
            print(f"        {title}:")
            print(f"          樣本數: {n_samples} | 原始分數平均: {mean_pre:.4f} (標準差: {std_pre:.4f}) | 校正分數平均: {mean_post:.4f} (標準差: {std_post:.4f})")
            print(f"          {subtitle}")
        else:
            subtitle = "Brier: N/A (無樣本)"
            print(f"        {title}: 無樣本")
            
        ax.hist(pre, bins=25, range=(0.0, 1.0), color='skyblue', alpha=0.5, label='Pre-cal (Raw)')
        ax.hist(post, bins=25, range=(0.0, 1.0), color='darkorange', alpha=0.6, label='Post-cal (Isotonic)')
        
        ax.set_title(f"{title}\n{subtitle}", fontsize=11, fontweight='bold')
        ax.set_xlim([-0.05, 1.05])
        ax.set_xlabel("信心分數 (Score)")
        ax.set_ylabel("次數 (Count)")
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc='upper center')
        
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    utils_calibration.setup_chinese_font()
    
    base_dir = "results/safety_guardrails_evaluation"
    os.makedirs(base_dir, exist_ok=True)
    sys.stdout = DualLogger(os.path.join(base_dir, "run_08_execution_log.txt"))
    
    print("="*60)
    print("LLM Safety Probe - 校正與直方圖分析腳本 (支援 y1, y2, y3 獨立任務與雙測試集交叉評估)")
    print("="*60)
    
    print("[1] 載入 Eval 資料集...")
    try:
        prep_eval = DataPreprocessor("experiment_results_eval.pkl")
        prep_eval.load_data()
        X_3d_eval = prep_eval.extract_features()
        y_targets_eval = prep_eval.create_targets()
        
        y1_eval = y_targets_eval[0].values if hasattr(y_targets_eval[0], 'values') else y_targets_eval[0]
        y3_eval = y_targets_eval[2].values if hasattr(y_targets_eval[2], 'values') else y_targets_eval[2]
    except Exception as e:
        print(f"錯誤: 無法讀取 Eval 資料集: {e}")
        return
        
    datasets = [
        ('data_aug', 'augmented_test1.pkl', 'augmented_test2.pkl', '資料增強'),
        ('data_align', 'aligned_test1.pkl', 'aligned_test2.pkl', '資料對齊')
    ]
    
    models = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
    targets_list = ['y1', 'y2', 'y3']
    
    for dataset_key, test1_file, test2_file, dataset_title in datasets:
        print(f"\n========================================\n處理資料組: {dataset_title} ({dataset_key})\n========================================\n")
        
        try:
            test1_data = joblib.load(test1_file)
            test2_data = joblib.load(test2_file)
        except Exception as e:
            print(f"錯誤: 無法讀取 {dataset_title} 資料集: {e}")
            continue
            
        for target_name in targets_list:
            print(f"\n----------------------------------------\n處理目標任務: {target_name.upper()}\n----------------------------------------")
            
            # Use target-specific test DataFrames
            df_test1 = test1_data[target_name]
            df_test2 = test2_data[target_name]
            
            # We also need the other split's test2 DataFrame for cross-evaluation
            if dataset_key == 'data_aug':
                other_test2_data = joblib.load('aligned_test2.pkl')
            else:
                other_test2_data = joblib.load('augmented_test2.pkl')
            df_test2_cross = other_test2_data[target_name]
            cross_name = 'aligned_test2' if dataset_key == 'data_aug' else 'augmented_test2'
            
            for layer_num in range(1, 7):
                print(f"\n[{layer_num}/6] 處理第 {layer_num} 層...")
                
                X_test1 = np.array(df_test1['hidden_state'].tolist())[:, layer_num - 1, :]
                X_test2 = np.array(df_test2['hidden_state'].tolist())[:, layer_num - 1, :]
                X_test2_cross = np.array(df_test2_cross['hidden_state'].tolist())[:, layer_num - 1, :]
                X_eval = X_3d_eval[:, layer_num - 1, :]
                
                y1_test1 = df_test1['y1'].values
                y3_test1 = df_test1['y3'].values
                
                y1_test2 = df_test2['y1'].values
                y3_test2 = df_test2['y3'].values
                
                y1_test2_cross = df_test2_cross['y1'].values
                y3_test2_cross = df_test2_cross['y3'].values
                
                for model_name in models:
                    model_path = f"results/unified_training/layer_{layer_num}/{model_name.lower()}_{target_name}_best.pkl"
                    
                    if not os.path.exists(model_path):
                        print(f"  - 跳過 {model_name} (找不到模型)")
                        continue
                        
                    clf = joblib.load(model_path)
                    
                    # Compute y_hat and gt variables dynamically based on target_name
                    if target_name == 'y1':
                        p_test1_prob = clf.predict_proba(X_test1)[:, 1]
                        y_hat_test1 = (p_test1_prob >= 0.5).astype(int)
                        
                        p_test2_prob = clf.predict_proba(X_test2)[:, 1]
                        y_hat_test2 = (p_test2_prob >= 0.5).astype(int)
                        
                        p_test2_cross_prob = clf.predict_proba(X_test2_cross)[:, 1]
                        y_hat_test2_cross = (p_test2_cross_prob >= 0.5).astype(int)
                        
                        p_eval_prob = clf.predict_proba(X_eval)[:, 1]
                        y_hat_eval = (p_eval_prob >= 0.5).astype(int)
                        
                        gt_test1, gt_test2, gt_test2_cross, gt_eval = y1_test1, y1_test2, y1_test2_cross, y1_eval
                    elif target_name == 'y2':
                        # 當評估 y2 向 y1_gt 靠攏時，使用 clf_y2 的預測，並對比 y1 真相
                        p_test1_prob = clf.predict_proba(X_test1)[:, 1]
                        y_hat_test1 = (p_test1_prob >= 0.5).astype(int)
                        
                        p_test2_prob = clf.predict_proba(X_test2)[:, 1]
                        y_hat_test2 = (p_test2_prob >= 0.5).astype(int)
                        
                        p_test2_cross_prob = clf.predict_proba(X_test2_cross)[:, 1]
                        y_hat_test2_cross = (p_test2_cross_prob >= 0.5).astype(int)
                        
                        p_eval_prob = clf.predict_proba(X_eval)[:, 1]
                        y_hat_eval = (p_eval_prob >= 0.5).astype(int)
                        
                        gt_test1, gt_test2, gt_test2_cross, gt_eval = y1_test1, y1_test2, y1_test2_cross, y1_eval
                    else: # y3 一致性任務
                        p_test1_prob = clf.predict_proba(X_test1)[:, 1]
                        y_hat_test1 = (p_test1_prob >= 0.5).astype(int)
                        
                        p_test2_prob = clf.predict_proba(X_test2)[:, 1]
                        y_hat_test2 = (p_test2_prob >= 0.5).astype(int)
                        
                        p_test2_cross_prob = clf.predict_proba(X_test2_cross)[:, 1]
                        y_hat_test2_cross = (p_test2_cross_prob >= 0.5).astype(int)
                        
                        p_eval_prob = clf.predict_proba(X_eval)[:, 1]
                        y_hat_eval = (p_eval_prob >= 0.5).astype(int)
                        
                        gt_test1, gt_test2, gt_test2_cross, gt_eval = y3_test1, y3_test2, y3_test2_cross, y3_eval
                    
                    # Predict raw probs for the target model
                    p_test1 = clf.predict_proba(X_test1)[:, 1]
                    p_test2 = clf.predict_proba(X_test2)[:, 1]
                    p_test2_cross = clf.predict_proba(X_test2_cross)[:, 1]
                    p_eval = clf.predict_proba(X_eval)[:, 1]
                    
                    # Construct calibration features
                    if target_name == 'y1' or target_name == 'y2':
                        score_pre_test1 = np.where(y1_test1 == 1, p_test1, 1.0 - p_test1)
                        score_pre_test2 = np.where(y1_test2 == 1, p_test2, 1.0 - p_test2)
                        score_pre_test2_cross = np.where(y1_test2_cross == 1, p_test2_cross, 1.0 - p_test2_cross)
                        score_pre_eval = np.where(y1_eval == 1, p_eval, 1.0 - p_eval)
                    else: # y3
                        score_pre_test1 = np.where(y3_test1 == 1, p_test1, 1.0 - p_test1)
                        score_pre_test2 = np.where(y3_test2 == 1, p_test2, 1.0 - p_test2)
                        score_pre_test2_cross = np.where(y3_test2_cross == 1, p_test2_cross, 1.0 - p_test2_cross)
                        score_pre_eval = np.where(y3_eval == 1, p_eval, 1.0 - p_eval)
                        
                    # Calibrate on test1
                    iso = IsotonicRegression(out_of_bounds='clip')
                    iso.fit(score_pre_test1, y3_test1)
                    
                    score_post_test1 = iso.predict(score_pre_test1)
                    score_post_test2 = iso.predict(score_pre_test2)
                    score_post_test2_cross = iso.predict(score_pre_test2_cross)
                    score_post_eval = iso.predict(score_pre_eval)
                    
                    # --- Test1 plot ---
                    save_path_test1 = f"results/safety_guardrails_evaluation/{dataset_key}/03_Quadrant_Histograms/{target_name}/test1/layer_{layer_num}/{model_name}_histogram.png"
                    plot_quadrant_histograms(
                        score_pre_test1, score_post_test1, y_hat_test1, gt_test1, y3_test1, 
                        layer_num, model_name, dataset_title, "Test1", target_name, save_path_test1
                    )
                    
                    # --- Test2 plot ---
                    test2_name_split = 'augmented_test2' if dataset_key == 'data_aug' else 'aligned_test2'
                    save_path_test2 = f"results/safety_guardrails_evaluation/{dataset_key}/03_Quadrant_Histograms/{target_name}/{test2_name_split}/layer_{layer_num}/{model_name}_histogram.png"
                    plot_quadrant_histograms(
                        score_pre_test2, score_post_test2, y_hat_test2, gt_test2, y3_test2, 
                        layer_num, model_name, dataset_title, test2_name_split.upper(), target_name, save_path_test2
                    )
                    
                    # --- Cross-Test2 plot ---
                    save_path_cross = f"results/safety_guardrails_evaluation/{dataset_key}/03_Quadrant_Histograms/{target_name}/{cross_name}/layer_{layer_num}/{model_name}_histogram.png"
                    plot_quadrant_histograms(
                        score_pre_test2_cross, score_post_test2_cross, y_hat_test2_cross, gt_test2_cross, y3_test2_cross, 
                        layer_num, model_name, dataset_title, cross_name.upper(), target_name, save_path_cross
                    )
                    
                    # --- Eval plot ---
                    save_path_eval = f"results/safety_guardrails_evaluation/{dataset_key}/03_Quadrant_Histograms/{target_name}/eval/layer_{layer_num}/{model_name}_histogram.png"
                    plot_quadrant_histograms(
                        score_pre_eval, score_post_eval, y_hat_eval, gt_eval, y3_eval, 
                        layer_num, model_name, dataset_title, "Eval", target_name, save_path_eval
                    )
                    
                    print(f"  - 輸出 {model_name} 直方圖完成 (test1, test2, {cross_name}, eval)")
                    
    print("\n直方圖 analysis 執行完畢！")

if __name__ == '__main__':
    main()
