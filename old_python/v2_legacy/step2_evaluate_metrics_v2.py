import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import utils_calibration

utils_calibration.setup_chinese_font()

def plot_metric_trend_v2(csv_path, metric_name, mode_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    eval_sets = df['eval_set'].unique()
    models = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
    colors = {'SGD': '#4C72B0', 'MLP': '#55A868', 'LGB': '#C44E52', 'LR': '#8172B3', 'RF': '#CCB974'}
    markers = {'SGD': 'o', 'MLP': 's', 'LGB': '^', 'LR': 'v', 'RF': 'D'}
    
    display_title = "Brier Score (越低越好)" if metric_name == 'brier' else "Log Loss (越低越好)"
    
    for eval_set in eval_sets:
        df_sub = df[df['eval_set'] == eval_set]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
        fig.suptitle(f"v2_20k 跨層 {display_title} 趨勢折線圖 - 評估集: {eval_set.upper()} ({mode_name})", fontsize=16, fontweight='bold')
        
        for idx, target_name in enumerate(['y1', 'y2', 'y3']):
            ax = axes[idx]
            df_target = df_sub[df_sub['task'] == target_name]
            
            for m in models:
                df_m = df_target[df_target['model'] == m].sort_values('layer')
                if len(df_m) > 0:
                    ax.plot(df_m['layer'], df_m[metric_name], marker=markers.get(m, 'o'),
                            color=colors.get(m, '#333333'), label=m, linewidth=2)
                            
            ax.set_title(f"任務: {target_name.upper()}", fontsize=13, fontweight='bold')
            ax.set_xlabel("隱藏層數 (Layer 1-6)", fontsize=11)
            if idx == 0: ax.set_ylabel(metric_name.capitalize(), fontsize=11)
            ax.set_xticks(range(1, 7))
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.legend(loc='upper right', fontsize=9)
            
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{metric_name}_trend_{eval_set}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

def main():
    base_dir = "results/v2_20k/02_Safety_Evaluation"
    modes = ["baseline", "split"]
    
    print("="*60)
    print("v2_20k 評估與校正 Pipeline - 步驟二: 可靠度圖表與指標趨勢")
    print("="*60)
    
    for mode in modes:
        print(f"\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n繪製模式圖表: {mode.upper()}\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        
        cache_dir = os.path.join(base_dir, "cache", mode)
        preds_path = os.path.join(cache_dir, "calibrated_predictions.pkl")
        csv_path = os.path.join(cache_dir, "all_metrics_records.csv")
        
        if not (os.path.exists(preds_path) and os.path.exists(csv_path)):
            print(f"警告: 模式 {mode.upper()} 的快取不存在，跳過。")
            continue
            
        predictions_cache = joblib.load(preds_path)
        
        # 1. 繪製 02_Metric_Trends
        trends_dir = os.path.join(base_dir, "02_Metric_Trends", mode)
        plot_metric_trend_v2(csv_path, 'brier', mode.upper(), trends_dir)
        plot_metric_trend_v2(csv_path, 'logloss', mode.upper(), trends_dir)
        print("  └─ [02_Metric_Trends] 完成！")
        
        # 2. 繪製 01_Reliability_Diagrams 與 04_Brier_Components (y1, y2, y3 放同一資料夾)
        rel_dir = os.path.join(base_dir, "01_Reliability_Diagrams", mode)
        comp_dir = os.path.join(base_dir, "04_Brier_Components", mode)
        os.makedirs(rel_dir, exist_ok=True)
        os.makedirs(comp_dir, exist_ok=True)
        
        for eval_set in ['test1', 'test2']:
            for target_name in ['y1', 'y2', 'y3']:
                for layer_num in range(1, 7):
                    cache_layer = predictions_cache[target_name].get(layer_num)
                    if not cache_layer: continue
                    
                    models_data = cache_layer['splits'][eval_set]
                    bin_edges_dict = cache_layer['bin_edges']
                    
                    title_prefix = f"v2_20k (Layer {layer_num} - {target_name.upper()} - {eval_set.upper()})"
                    
                    # Reliability Line
                    line_path = os.path.join(rel_dir, f"layer_{layer_num}_{eval_set}_{target_name}_line.png")
                    utils_calibration.plot_comparison_line(models_data, bin_edges_dict, f"{title_prefix} 校正對比線", line_path)
                    
                    # Reliability Bars
                    bar_path = os.path.join(rel_dir, f"layer_{layer_num}_{eval_set}_{target_name}_bars.png")
                    utils_calibration.plot_side_by_side_bars(models_data, bin_edges_dict, f"{title_prefix} 並列可靠度柱狀圖", bar_path)
                    
                    # Split Y
                    split_y_path = os.path.join(rel_dir, f"layer_{layer_num}_{eval_set}_{target_name}_split_y.png")
                    utils_calibration.plot_comparison_line_split_y(models_data, bin_edges_dict, f"{title_prefix} y1 分拆校正對比線", split_y_path)
                    
                    # Brier Components
                    rel_res_path = os.path.join(comp_dir, f"layer_{layer_num}_{eval_set}_{target_name}_brier_rel_res_bar.png")
                    utils_calibration.plot_brier_components_bar(models_data, bin_edges_dict, f"{title_prefix} Brier Rel/Res 貢獻", rel_res_path)
                    
                    weight_path = os.path.join(comp_dir, f"layer_{layer_num}_{eval_set}_{target_name}_brier_bin_weights_bar.png")
                    utils_calibration.plot_bin_weights_bar(models_data, bin_edges_dict, f"{title_prefix} Brier Bin 樣本佔比", weight_path)
                    
        print(f"  └─ [01_Reliability_Diagrams] 與 [04_Brier_Components] 完成！")

if __name__ == '__main__':
    main()
