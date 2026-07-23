"""
evaluation_pipeline/step4_generate_custom_grids.py
===================================================
綜合矩陣圖與自訂指標評估繪圖模組 (Custom Grid & Advanced Visualization Module)

包含：
1. 3x3 跨指標趨勢矩陣圖 (Brier Score, Log Loss)
2. 3x6 Split y1 可靠度矩陣圖 (按任務與按模型)
3. 2x3 雙 Y 軸 Brier 組分圖檔生成器 (dual_y / single_y)
4. Native Bins 與 Adaptive Bins 3x6 可靠度圖檔生成器
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import utils_calibration

utils_calibration.setup_chinese_font()

BASE_DIR = r"C:\Users\weiwe\OneDrive\Desktop\Safety-training dataset\results\safety_guardrails_evaluation"
DATA_ALIGN_DIR = os.path.join(BASE_DIR, "data_align", "split")
CACHE_PATH = os.path.join(BASE_DIR, "cache", "split", "calibrated_predictions.pkl")
CSV_PATH = os.path.join(BASE_DIR, "cache", "split", "all_metrics_records.csv")

TARGETS = ['y1', 'y2', 'y3']
MODELS = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
COLORS = {'SGD': '#4C72B0', 'MLP': '#55A868', 'LGB': '#C44E52', 'LR': '#8172B3', 'RF': '#CCB974'}
MARKERS = {'SGD': 'o', 'MLP': 's', 'LGB': '^', 'LR': 'v', 'RF': 'D'}

TARGET_TITLES = {
    'y1': 'Y1 (模型回應安全性)',
    'y2': 'Y2 (提示詞有害性)',
    'y3': 'Y3 (安全判定一致性)'
}

SPLIT_DISPLAY_NAMES = {
    'test1': 'Aligned Test 1',
    'test2': 'Aligned Test 2',
    'eval': 'Eval (外部評估集)'
}

def generate_3x3_metrics_trends():
    """產出 Brier Score 與 Log Loss 3x3 矩陣走勢圖"""
    print("\n[Step 4.1] 繪製 3x3 跨指標趨勢矩陣圖 (Brier & Log Loss)...")
    if not os.path.exists(CSV_PATH):
        print(f"警告: 找不到指標紀錄 CSV: {CSV_PATH}")
        return
        
    df = pd.read_csv(CSV_PATH)
    df_align = df[df['data_group'] == 'Data Align'].copy()
    out_dir = os.path.join(DATA_ALIGN_DIR, "01_Metrics_Trends")
    os.makedirs(out_dir, exist_ok=True)
    
    cols = ['test1', 'test2', 'eval']
    
    for metric_key, metric_title, filename in [
        ('brier', 'Brier Score (越低越好)', 'brier_score_trend_data_align_split_3x3_grid.png'),
        ('logloss', 'Log Loss (越低越好)', 'log_loss_trend_data_align_split_3x3_grid.png')
    ]:
        fig, axes = plt.subplots(3, 3, figsize=(18, 14), sharex=True)
        fig.suptitle(f'跨隱藏層 (Layer 1~6) {metric_title} 走勢圖 3x3 矩陣對比\n(模式: Split 校正 | 資料對齊組)', 
                     fontsize=18, fontweight='bold', y=0.98)
        
        for r_idx, task in enumerate(TARGETS):
            for c_idx, eval_set in enumerate(cols):
                ax = axes[r_idx, c_idx]
                cell_title = f"{TARGET_TITLES[task]} - {SPLIT_DISPLAY_NAMES.get(eval_set, eval_set)}"
                ax.set_title(cell_title, fontsize=12, fontweight='bold', pad=8)
                
                sub_df = df_align[(df_align['task'] == task) & (df_align['eval_set'] == eval_set)]
                for m_name in MODELS:
                    m_df = sub_df[sub_df['model'] == m_name].sort_values('layer')
                    if len(m_df) > 0:
                        ax.plot(m_df['layer'].values, m_df[metric_key].values,
                                marker=MARKERS.get(m_name, 'o'), color=COLORS.get(m_name, '#333333'),
                                label=m_name, linewidth=2, markersize=7)
                                
                ax.set_xticks([1, 2, 3, 4, 5, 6])
                ax.set_xticklabels(['L1', 'L2', 'L3', 'L4', 'L5', 'L6'], fontsize=10)
                ax.grid(True, linestyle='--', alpha=0.4)
                if c_idx == 0: ax.set_ylabel(metric_title, fontsize=11, fontweight='bold')
                if r_idx == 2: ax.set_xlabel('特徵層 (Layer)', fontsize=11, fontweight='bold')
                ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
                
        plt.tight_layout()
        fig.subplots_adjust(top=0.92)
        out_path = os.path.join(out_dir, filename)
        plt.savefig(out_path, dpi=180, bbox_inches='tight')
        plt.close()
        print(f"  └─ 成功儲存 {metric_title} 3x3 矩陣圖至: {out_path}")

def main():
    print("="*70)
    print("執行 Step 4: 綜合矩陣拼圖與高級可視化產出")
    print("="*70)
    generate_3x3_metrics_trends()
    print("Step 4 處理完畢！")

if __name__ == '__main__':
    main()
