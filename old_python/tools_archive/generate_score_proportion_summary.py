"""
生成 test1 與 test2 下 y1=0 與 y1=1 修正前後分數比例分布對比圖 (Score Proportion Summary)
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils_calibration

utils_calibration.setup_chinese_font()

def generate_proportion_summary_figure(cache_path, target='y1', layer=1, model='LGB', output_path='results/plots/score_proportion_summary_test1_test2.png'):
    if not os.path.exists(cache_path):
        print(f"錯誤: 找不到預測快取 {cache_path}")
        return
        
    predictions_cache = joblib.load(cache_path)
    layer_data = predictions_cache[target][layer]['splits']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"test1 & test2 修正前後分數比例分布對比 (y1=0 vs y1=1)\n任務: {target.upper()} | 模型: {model} | 層數: Layer {layer}", 
                 fontsize=15, fontweight='bold', y=0.98)
                 
    splits = ['test1', 'test2']
    groups = [(0, 'Guardrail 放行 (y1=0 / Safe)'), (1, 'Guardrail 攔截 (y1=1 / Unsafe)')]
    
    colors_pre = '#B07A4C'
    colors_post = '#4C72B0'
    
    bins = np.linspace(0.0, 1.0, 21)  # 20 個比例 bins
    
    for row_idx, split in enumerate(splits):
        data_split = layer_data[split][model]
        score_pre = np.array(data_split['score_pre'])
        y_prob_cal = np.array(data_split['y_prob'])
        y1_labels = np.array(data_split['y1'])
        y_true = np.array(data_split['y_true'])
        
        for col_idx, (group_val, group_label) in enumerate(groups):
            ax = axes[row_idx, col_idx]
            mask = (y1_labels == group_val)
            
            if np.sum(mask) == 0:
                ax.set_title(f"{split.upper()} | {group_label} (無樣本)", fontsize=12)
                continue
                
            pre_g = score_pre[mask]
            post_g = y_prob_cal[mask]
            n_g = len(pre_g)
            
            # 計算樣本比例權重 (%)
            w_pre = np.ones(n_g) / n_g * 100.0
            w_post = np.ones(n_g) / n_g * 100.0
            
            ax.hist(pre_g, bins=bins, weights=w_pre, color=colors_pre, alpha=0.45, 
                    edgecolor='black', linewidth=0.5, label='修正前 (Raw Score)')
            ax.hist(post_g, bins=bins, weights=w_post, color=colors_post, alpha=0.60, 
                    edgecolor='black', linewidth=0.5, label='修正後 (Isotonic Probability)')
                    
            metrics_raw = utils_calibration.calculate_all_metrics(y_true[mask], pre_g)
            metrics_cal = utils_calibration.calculate_all_metrics(y_true[mask], post_g)
            
            sub_title = (f"【{split.upper()}】{group_label} (樣本數: {n_g})\n"
                         f"Raw Brier: {metrics_raw['brier']:.4f} -> Cal Brier: {metrics_cal['brier']:.4f}")
                         
            ax.set_title(sub_title, fontsize=11, fontweight='bold')
            ax.set_xlim([-0.05, 1.05])
            ax.set_xlabel("分數 / 機率 (Score / Probability)", fontsize=10, fontweight='bold')
            ax.set_ylabel("該組別樣本比例 (%)", fontsize=10, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.legend(loc='upper right', fontsize=9)
            
    plt.tight_layout()
    fig.subplots_adjust(top=0.91)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] 修正前後分數比例分布對比圖已產出: {output_path}")

def main():
    cache_path = "results/safety_guardrails_evaluation/cache/calibrated_predictions.pkl"
    generate_proportion_summary_figure(cache_path, target='y1', layer=1, model='LGB', 
                                       output_path='results/plots/score_proportion_summary_y1_lgb.png')
    generate_proportion_summary_figure(cache_path, target='y1', layer=1, model='SGD', 
                                       output_path='results/plots/score_proportion_summary_y1_sgd.png')

if __name__ == '__main__':
    main()
