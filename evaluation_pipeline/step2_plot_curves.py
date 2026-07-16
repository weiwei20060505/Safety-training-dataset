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

def generate_metrics_line_plot(df, dataset_title, save_path):
    """
    Plots a grid comparing SGD, MLP, LGB, LR, RF across layers.
    Columns: Brier Score, Log Loss
    """
    utils_calibration.setup_chinese_font()
    models = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
    colors = {'SGD': '#4C72B0', 'MLP': '#55A868', 'LGB': '#C44E52', 'LR': '#8172B3', 'RF': '#CCB974'}
    markers = {'SGD': 'o', 'MLP': 's', 'LGB': '^', 'LR': 'v', 'RF': 'D'}
    
    metrics = ['Brier Score', 'Log Loss']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    fig.suptitle(f'指標隨層數變化趨勢 - {dataset_title}', fontsize=16, fontweight='bold', y=1.02)
    
    for c_idx, mt in enumerate(metrics):
        ax = axes[c_idx]
        df_sub = df[df['metric'] == mt]
        
        best_val = float('inf')
        best_model = None
        best_layer = None
        
        for model in models:
            df_model = df_sub[df_sub['model'] == model].sort_values('layer')
            if df_model.empty:
                continue
                
            layers = df_model['layer'].values
            vals = df_model['value'].values
            
            ax.plot(layers, vals, label=model, color=colors[model], 
                    marker=markers[model], markersize=6, linewidth=1.5, alpha=0.9)
            
            # Track best
            min_idx = np.argmin(vals)
            if vals[min_idx] < best_val:
                best_val = vals[min_idx]
                best_model = model
                best_layer = layers[min_idx]
                
        # Highlight best model
        if best_model is not None:
            ax.plot(best_layer, best_val, marker='*', color='#FFD700', 
                    markersize=14, markeredgecolor='black', markeredgewidth=1, 
                    zorder=5, label='最佳模型')
                    
        ax.set_title(f'{mt}', fontsize=14, fontweight='bold')
        ax.set_ylabel('指標值 (越低越好)', fontsize=12)
        ax.set_xlabel('特徵層數 (Layer)', fontsize=12)
            
        num_layers = int(df['layer'].max()) if not df.empty else 6
        ax.set_xticks(range(1, num_layers + 1))
        ax.grid(True, linestyle='--', alpha=0.3)
        
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=10)
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="LLM Safety Probe Pipeline - Step 2: Plot Calibration Curves & Metrics")
    parser.add_argument("--mode", type=str, choices=["baseline", "split", "all"], default="all",
                        help="Calibration mode of cache to plot: baseline, split, or all")
    args = parser.parse_args()
    
    base_dir = "results/safety_guardrails_evaluation"
    
    print("="*60)
    print(f"LLM Safety Probe Pipeline - 步驟二: 輕量視覺化 (模式: {args.mode.upper()})")
    print("="*60)
    
    mode_titles = {'baseline': '(傳統單一校正)', 'split': '(條件雙軌拆分校正)'}
    
    # Determine modes to run
    modes_to_run = ["baseline", "split"] if args.mode == "all" else [args.mode]
    
    for mode in modes_to_run:
        print(f"\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n繪製校正模式圖表: {mode.upper()}\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        
        cache_dir = os.path.join(base_dir, "cache", mode)
        metrics_path = os.path.join(cache_dir, "all_metrics_records.csv")
        preds_path = os.path.join(cache_dir, "calibrated_predictions.pkl")
        
        if not os.path.exists(metrics_path) or not os.path.exists(preds_path):
            print(f"警告: 模式 {mode.upper()} 的快取檔案不存在，跳過此模式的繪圖。")
            continue
            
        # 💡 [陷阱二解決策略] 在迴圈內部載入當前 mode 的快取數據，防止讀取錯誤或重複數據！
        df_metrics = pd.read_csv(metrics_path)
        predictions_cache = joblib.load(preds_path)
        
        datasets = [
            ('data_aug', '資料增強'),
            ('data_align', '資料對齊')
        ]
        targets_list = ['y1', 'y2', 'y3']
        
        print("[1] 繪製特徵層跨模型指標比較圖 (Metrics Trends)...")
        for dataset_key, dataset_title in datasets:
            dg_name = 'Data Aug' if dataset_key == 'data_aug' else 'Data Align'
            df_group = df_metrics[df_metrics['data_group'] == dg_name]
            
            # Unique splits in this group
            splits = df_group['eval_set'].unique()
            
            for split_name in splits:
                for target_name in targets_list:
                    df_sub = df_group[(df_group['eval_set'] == split_name) & (df_group['task'] == target_name)]
                    if df_sub.empty:
                        continue
                    
                    # Format for generate_metrics_line_plot
                    records = []
                    for _, row in df_sub.iterrows():
                        records.append({'layer': row['layer'], 'model': row['model'], 'metric': 'Brier Score', 'value': row['brier']})
                        records.append({'layer': row['layer'], 'model': row['model'], 'metric': 'Log Loss', 'value': row['logloss']})
                    
                    df_plot = pd.DataFrame(records)
                    
                    # 💡 [陷阱三解決策略] 使用動態路徑與標題對齊
                    trend_path = f"{base_dir}/{dataset_key}/{mode}/01_Metrics_Trends/{split_name}/{target_name}_metrics_trend.png"
                    title_text = f"{dataset_title} - {split_name.upper()} - 任務: {target_name.upper()} {mode_titles[mode]}"
                    
                    generate_metrics_line_plot(df_plot, title_text, trend_path)
                    
        print("[2] 繪製可靠度曲線與 Brier 分解指標柱狀圖...")
        for dataset_key, dataset_title in datasets:
            for target_name in targets_list:
                for layer_num in range(1, 7):
                    cache_layer = predictions_cache[dataset_key][target_name].get(layer_num)
                    if not cache_layer:
                        continue
                    
                    bin_edges_dict = cache_layer['bin_edges']
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
                        
                        # 💡 [陷阱三解決策略] 建立對稱的輸出路徑
                        mode_subfolder = f"{base_dir}/{dataset_key}/{mode}"
                        
                        # 1. Plot Reliability Curves
                        rel_path = f"{mode_subfolder}/02_Reliability_Curves/{eval_set_name}/layer_{layer_num}/{target_name}_reliability.png"
                        utils_calibration.plot_comparison_line(
                            models_data, bin_edges_dict,
                            f"第 {layer_num} 層可靠度曲線 ({dataset_title}) {mode_titles[mode]} - {eval_set_name.upper()} - 任務: {target_name.upper()}", rel_path
                        )
                        
                        # 2. Plot Reliability Curves split by Y
                        rel_split_path = f"{mode_subfolder}/02_Reliability_Curves_split_y/{eval_set_name}/layer_{layer_num}/{target_name}_reliability_split.png"
                        utils_calibration.plot_comparison_line_split_y(
                            models_data, bin_edges_dict,
                            f"第 {layer_num} 層可靠度曲線 ({dataset_title} - 依 y_i 拆分) {mode_titles[mode]} - {eval_set_name.upper()} - 任務: {target_name.upper()}", rel_split_path
                        )
                        
                        # 3. Plot Brier Components (Rel and Res only)
                        brier_bar_path = f"{mode_subfolder}/04_Brier_Components/{eval_set_name}/layer_{layer_num}/{target_name}_brier_components.png"
                        utils_calibration.plot_brier_components_bar(
                            models_data, bin_edges_dict,
                            f"第 {layer_num} 層分區間 Brier 分解指標柱狀圖 ({dataset_title}) {mode_titles[mode]} - {eval_set_name.upper()} - 任務: {target_name.upper()}", brier_bar_path
                        )
                        
                        # 4. Plot Bin Weights (Sample proportions)
                        weights_bar_path = f"{mode_subfolder}/04_Brier_Components/{eval_set_name}/layer_{layer_num}/{target_name}_bin_weights.png"
                        utils_calibration.plot_bin_weights_bar(
                            models_data, bin_edges_dict,
                            f"第 {layer_num} 層樣本佔比柱狀圖 ({dataset_title}) {mode_titles[mode]} - {eval_set_name.upper()} - 任務: {target_name.upper()}", weights_bar_path
                        )

        print(f"模式 {mode.upper()} 繪圖完畢！所有圖表已寫入 results/.../{mode}/ 資料夾。")
        
    print("\n所有指定模式圖表皆已繪製完成！")

if __name__ == '__main__':
    main()
