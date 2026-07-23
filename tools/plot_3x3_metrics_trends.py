import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import utils_calibration

utils_calibration.setup_chinese_font()

csv_path = r"C:\Users\weiwe\OneDrive\Desktop\Safety-training dataset\results\safety_guardrails_evaluation\cache\split\all_metrics_records.csv"
df = pd.read_csv(csv_path)

# Filter for Data Align group
df_align = df[df['data_group'] == 'Data Align'].copy()

base_out = r"C:\Users\weiwe\OneDrive\Desktop\Safety-training dataset\results\safety_guardrails_evaluation\data_align\split\01_Metrics_Trends"
os.makedirs(base_out, exist_ok=True)

rows = ['y1', 'y2', 'y3']
cols = ['test1', 'test2', 'eval']
models_list = ['SGD', 'MLP', 'LGB', 'LR', 'RF']

colors = {'SGD': '#4C72B0', 'MLP': '#55A868', 'LGB': '#C44E52', 'LR': '#8172B3', 'RF': '#CCB974'}
markers = {'SGD': 'o', 'MLP': 's', 'LGB': '^', 'LR': 'v', 'RF': 'D'}

target_display_names = {
    'y1': 'Y1 (模型回應安全性)',
    'y2': 'Y2 (提示詞有害性)',
    'y3': 'Y3 (安全判定一致性)'
}

col_display_names = {
    'test1': 'Test 1 (Aligned)',
    'test2': 'Test 2 (Aligned)',
    'eval': 'Eval (外部評估集)'
}

def plot_3x3_grid(metric_name, metric_title, filename):
    fig, axes = plt.subplots(3, 3, figsize=(18, 14), sharex=True)
    fig.suptitle(f'跨隱藏層 (Layer 1~6) {metric_title} 走勢圖 3x3 矩陣對比\n(模式: Split 校正 | 資料對齊組)', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    for r_idx, task in enumerate(rows):
        for c_idx, eval_set in enumerate(cols):
            ax = axes[r_idx, c_idx]
            
            # Subtitle for each cell
            cell_title = f"{target_display_names[task]} - {col_display_names[eval_set]}"
            ax.set_title(cell_title, fontsize=12, fontweight='bold', pad=8)
            
            # Filter data
            sub_df = df_align[(df_align['task'] == task) & (df_align['eval_set'] == eval_set)]
            
            for model_name in models_list:
                m_df = sub_df[sub_df['model'] == model_name].sort_values('layer')
                if len(m_df) > 0:
                    layers = m_df['layer'].values
                    vals = m_df[metric_name].values
                    ax.plot(layers, vals, marker=markers.get(model_name, 'o'),
                            color=colors.get(model_name, '#333333'),
                            label=model_name, linewidth=2, markersize=7)
                            
            ax.set_xticks([1, 2, 3, 4, 5, 6])
            ax.set_xticklabels(['L1', 'L2', 'L3', 'L4', 'L5', 'L6'], fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.4)
            
            if c_idx == 0:
                ax.set_ylabel(f'{metric_title}', fontsize=11, fontweight='bold')
            if r_idx == 2:
                ax.set_xlabel('特徵層 (Layer)', fontsize=11, fontweight='bold')
                
            # Add legend in top-right cell or each cell
            ax.legend(loc='upper right', fontsize=8, framealpha=0.8)

    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    
    out_path = os.path.join(base_out, filename)
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Successfully saved {metric_title} 3x3 grid to: {out_path}")

def main():
    plot_3x3_grid('brier', 'Brier Score (越低越好)', 'brier_score_3x3_grid.png')
    plot_3x3_grid('logloss', 'Log Loss (越低越好)', 'log_loss_3x3_grid.png')

if __name__ == '__main__':
    main()
