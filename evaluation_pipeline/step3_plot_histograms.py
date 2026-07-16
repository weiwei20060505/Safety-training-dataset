import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent directory to sys.path to import core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils_calibration

def plot_quadrant_histograms(pre_scores, post_scores, y_hat, y_gt, y3_gt, layer_num, model_name, dataset_title, split_name, target_name, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"第 {layer_num} 層隱藏狀態 - {model_name} 校正前後分數分佈\n({dataset_title} - {split_name} - 任務: {target_name.upper()})", fontsize=16, fontweight='bold')
    
    # Groups definition: (y_hat, y_gt)
    if target_name in ['y1', 'y2']:
        groups = [
            ((0, 0), "Group 1: Guardrail Safe (0) | GT Safe (0)", axes[0, 0]),
            ((0, 1), "Group 2: Guardrail Safe (0) | GT Unsafe (1)", axes[0, 1]),
            ((1, 0), "Group 3: Guardrail Unsafe (1) | GT Safe (0)", axes[1, 0]),
            ((1, 1), "Group 4: Guardrail Unsafe (1) | GT Unsafe (1)", axes[1, 1]),
        ]
    else:  # y3
        groups = [
            ((0, 0), "Group 1: Pred Inconsistent (0) | GT Inconsistent (0)", axes[0, 0]),
            ((0, 1), "Group 2: Pred Inconsistent (0) | GT Consistent (1)", axes[0, 1]),
            ((1, 0), "Group 3: Pred Consistent (1) | GT Inconsistent (0)", axes[1, 0]),
            ((1, 1), "Group 4: Pred Consistent (1) | GT Consistent (1)", axes[1, 1]),
        ]
        
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
    parser = argparse.ArgumentParser(description="LLM Safety Probe Pipeline - Step 3: Quadrant Histograms")
    parser.add_argument("--mode", type=str, choices=["baseline", "split", "all"], default="all",
                        help="Calibration mode of cache to plot histograms: baseline, split, or all")
    args = parser.parse_args()
    
    utils_calibration.setup_chinese_font()
    base_dir = "results/safety_guardrails_evaluation"
    
    print("="*60)
    print(f"LLM Safety Probe Pipeline - 步驟三: 輕量視覺化 (模式: {args.mode.upper()})")
    print("="*60)
    
    mode_titles = {'baseline': '(傳統單一校正)', 'split': '(條件雙軌拆分校正)'}
    
    # Determine modes to run
    modes_to_run = ["baseline", "split"] if args.mode == "all" else [args.mode]
    
    for mode in modes_to_run:
        print(f"\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n繪製校正模式直方圖: {mode.upper()}\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        
        preds_path = os.path.join(base_dir, "cache", mode, "calibrated_predictions.pkl")
        
        if not os.path.exists(preds_path):
            print(f"警告: 模式 {mode.upper()} 的快取檔案不存在，跳過此模式的直方圖。")
            continue
            
        # 💡 [陷阱二解決策略] 在迴圈內部載入當前 mode 的快取數據，防止讀取錯誤或重複數據！
        predictions_cache = joblib.load(preds_path)
        
        datasets = [
            ('data_aug', '資料增強'),
            ('data_align', '資料對齊')
        ]
        targets_list = ['y1', 'y2', 'y3']
        models_list = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
        
        print("[1] 開始繪製直方圖...")
        for dataset_key, dataset_title in datasets:
            for target_name in targets_list:
                for layer_num in range(1, 7):
                    cache_layer = predictions_cache[dataset_key][target_name].get(layer_num)
                    if not cache_layer:
                        continue
                    
                    splits_dict = cache_layer['splits']
                    
                    for split_name, models_data in splits_dict.items():
                        # Map test2_cross key to its actual test2 filename string for saving
                        if split_name == 'test2_cross':
                            eval_set_name = 'aligned_test2' if dataset_key == 'data_aug' else 'augmented_test2'
                        else:
                            eval_set_name = 'augmented_test2' if (split_name == 'test2' and dataset_key == 'data_aug') else \
                                            'aligned_test2' if (split_name == 'test2' and dataset_key == 'data_align') else \
                                            'augmented_test1' if (split_name == 'test1' and dataset_key == 'data_aug') else \
                                            'aligned_test1' if (split_name == 'test1' and dataset_key == 'data_align') else \
                                            split_name
                                            
                        for model_name in models_list:
                            data = models_data.get(model_name)
                            if not data:
                                continue
                                
                            # Reconstruct predictions and variables
                            y1 = data['y1']
                            y3 = data['y3']
                            y_prob_pre = data['y_prob_pre']
                            y_prob_post = data['y_prob']
                            
                            # Reconstruct y_hat (pre-cal prediction thresholded at 0.5)
                            y_hat = (y_prob_pre >= 0.5).astype(int)
                            
                            # 💡 修正內部生成 score_pre 與 y_gt 的邏輯，避免 Y3 任務反轉
                            if target_name in ['y1', 'y2']:
                                score_pre = np.where(y1 == 1, y_prob_pre, 1.0 - y_prob_pre)
                                y_gt = y1
                            else:  # y3 任務：嚴格禁止轉換！
                                score_pre = y_prob_pre 
                                y_gt = y3
                                
                            # 💡 [陷阱三解決策略] 建立對稱的輸出路徑與標題
                            save_path = f"{base_dir}/{dataset_key}/{mode}/03_Quadrant_Histograms/{target_name}/{eval_set_name}/layer_{layer_num}/{model_name}_histogram.png"
                            title_text = f"{dataset_title} {mode_titles[mode]}"
                            
                            plot_quadrant_histograms(
                                score_pre, y_prob_post, y_hat, y_gt, y3,
                                layer_num, model_name, title_text, eval_set_name.upper(), target_name, save_path
                            )
                            print(f"  - 輸出 {model_name} 直方圖完成 ({eval_set_name} | {mode.upper()})")
                            
        print(f"模式 {mode.upper()} 直方圖繪製完畢！")

if __name__ == '__main__':
    main()
