import os
import re
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def setup_chinese_font():
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'PMingLiU', 'DFKai-SB', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False 

def parse_log_file(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
        
    data = []
    
    # Regular expressions to parse calibration_metrics_log.txt
    layer_target_pat = re.compile(r"===== \[第\s*(\d+)\s*層\s*-\s*([^\]]+)\] Isotonic 校正與指標評估 =====")
    model_pat = re.compile(r"\s*\[模型\s*([A-Za-z0-9_]+)\s*評估報告\]")
    dataset_pat = re.compile(r"\s*├─\s*\[([^\]]+)\s*資料集評估\]")
    brier_pat = re.compile(r"Brier Score\s*:\s*([\d\.]+)\s*➔\s*([\d\.]+)")
    ece_pat = re.compile(r"ECE\s*\(校正誤差\):\s*([\d\.]+)\s*➔\s*([\d\.]+)")
    logloss_pat = re.compile(r"Log Loss\s*:\s*([\d\.]+)\s*➔\s*([\d\.]+)")
    
    current_layer = None
    current_target = None
    current_model = None
    current_dataset = None
    
    target_map = {
        "模型回覆安全性預測": "y1",
        "提示詞有害性預測": "y2",
        "安全判定一致性預測": "y3"
    }
    
    dataset_map = {
        "Test1 (校正用)": "test1",
        "Test2 (未見過)": "test2",
        "Eval  (OOD集)": "eval"
    }

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            
            # Check for layer and target
            m = layer_target_pat.match(line_str)
            if m:
                current_layer = int(m.group(1))
                target_str = m.group(2).strip()
                current_target = target_map.get(target_str, target_str)
                continue
                
            # Check for model
            m = model_pat.match(line_str)
            if m:
                current_model = m.group(1).strip()
                continue
                
            # Check for dataset
            m = dataset_pat.match(line_str)
            if m:
                ds_str = m.group(1).strip()
                current_dataset = dataset_map.get(ds_str, ds_str)
                continue
                
            # Check for metrics
            m_b = brier_pat.search(line_str)
            if m_b:
                raw_val = float(m_b.group(1))
                cal_val = float(m_b.group(2))
                data.append({
                    "layer": current_layer,
                    "target": current_target,
                    "model": current_model,
                    "dataset": current_dataset,
                    "metric": "Brier Score",
                    "raw": raw_val,
                    "calibrated": cal_val
                })
                continue
                
            m_e = ece_pat.search(line_str)
            if m_e:
                raw_val = float(m_e.group(1))
                cal_val = float(m_e.group(2))
                data.append({
                    "layer": current_layer,
                    "target": current_target,
                    "model": current_model,
                    "dataset": current_dataset,
                    "metric": "ECE",
                    "raw": raw_val,
                    "calibrated": cal_val
                })
                continue
                
            m_l = logloss_pat.search(line_str)
            if m_l:
                raw_val = float(m_l.group(1))
                cal_val = float(m_l.group(2))
                data.append({
                    "layer": current_layer,
                    "target": current_target,
                    "model": current_model,
                    "dataset": current_dataset,
                    "metric": "Log Loss",
                    "raw": raw_val,
                    "calibrated": cal_val
                })
                continue
                
    return pd.DataFrame(data)

def generate_comparison_plots(df, output_dir):
    setup_chinese_font()
    os.makedirs(output_dir, exist_ok=True)
    
    models = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
    colors = {'SGD': '#4C72B0', 'MLP': '#55A868', 'LGB': '#C44E52', 'LR': '#8172B3', 'RF': '#CCB974'}
    markers = {'SGD': 'o', 'MLP': 's', 'LGB': '^', 'LR': 'v', 'RF': 'D'}
    
    targets = ['y1', 'y2', 'y3']
    target_names = {
        'y1': 'y1 (模型回覆安全性)',
        'y2': 'y2 (提示詞有害性)',
        'y3': 'y3 (安全判定一致性)'
    }
    
    metrics = ['Brier Score', 'ECE', 'Log Loss']
    datasets = ['test1', 'test2', 'eval']
    dataset_titles = {
        'test1': 'Test1 (校正用訓練集)',
        'test2': 'Test2 (獨立測試集 - 同分佈)',
        'eval': 'Eval (OOD 外部驗證集)'
    }
    
    summary_data = []

    for ds in datasets:
        df_ds = df[df['dataset'] == ds]
        if df_ds.empty:
            continue
            
        fig, axes = plt.subplots(3, 3, figsize=(18, 15), sharex=True)
        fig.suptitle(f'校正後模型多維度指標對比 - {dataset_titles[ds]}', fontsize=20, fontweight='bold', y=0.96)
        
        for r_idx, tg in enumerate(targets):
            for c_idx, mt in enumerate(metrics):
                ax = axes[r_idx, c_idx]
                
                # Filter data for this subplot
                df_sub = df_ds[(df_ds['target'] == tg) & (df_ds['metric'] == mt)]
                
                best_val = float('inf')
                best_model = None
                best_layer = None
                
                # Plot line for each model
                for model in models:
                    df_model = df_sub[df_sub['model'] == model].sort_values('layer')
                    if df_model.empty:
                        continue
                    
                    layers = df_model['layer'].values
                    vals = df_model['calibrated'].values
                    
                    ax.plot(layers, vals, label=model, color=colors[model], 
                            marker=markers[model], markersize=6, linewidth=1.5, alpha=0.9)
                    
                    # Track the minimum
                    min_idx = np.argmin(vals)
                    if vals[min_idx] < best_val:
                        best_val = vals[min_idx]
                        best_model = model
                        best_layer = layers[min_idx]
                
                # Highlight the best point in the subplot
                if best_model is not None:
                    ax.plot(best_layer, best_val, marker='*', color='#FFD700', 
                            markersize=14, markeredgecolor='black', markeredgewidth=1, 
                            zorder=5, label='最佳模型')
                    
                    # Store in summary list
                    summary_data.append({
                        'dataset': ds,
                        'target': tg,
                        'metric': mt,
                        'best_model': best_model,
                        'best_layer': best_layer,
                        'best_val': best_val
                    })
                
                # Title and labels
                if r_idx == 0:
                    ax.set_title(f'{mt}', fontsize=14, fontweight='bold')
                if c_idx == 0:
                    ax.set_ylabel(f'{target_names[tg]}\n指標值 (越低越好)', fontsize=12, fontweight='bold')
                if r_idx == 2:
                    ax.set_xlabel('特徵層數 (Layer)', fontsize=12)
                    
                ax.set_xticks(range(1, 7))
                ax.grid(True, linestyle='--', alpha=0.5)
                
                # Avoid showing duplicate legend entries
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=9)
        
        plt.tight_layout()
        fig.subplots_adjust(top=0.90)
        
        # Save output image
        img_path = os.path.join(output_dir, f'calibration_comparison_{ds}.png')
        fig.savefig(img_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[繪圖完成] 已儲存對比圖至: {img_path}")
        
    return pd.DataFrame(summary_data)

def main():
    log_filepath = "results/reliability_diagrams/calibration_metrics_log.txt"
    output_dir = "results/reliability_diagrams"
    
    print("=" * 60)
    print(" 啟動 Isotonic 矯正後模型多維度指標對比繪圖")
    print("=" * 60)
    
    # 1. Parse log file
    df = parse_log_file(log_filepath)
    if df is None or df.empty:
        print("錯誤: 無法解析指標日誌檔，或資料為空！")
        sys.exit(1)
    print(f"成功解析共 {len(df)} 筆指標數據。")
    
    # 2. Generate plots
    df_summary = generate_comparison_plots(df, output_dir)
    
    # 3. Print best summary report
    print("\n" + "=" * 60)
    print(" 各目標、各資料集在校正後的「最佳模型與特徵層」摘要報告")
    print("=" * 60)
    
    target_display = {
        'y1': 'y1 (模型回覆安全性)',
        'y2': 'y2 (提示詞有害性)',
        'y3': 'y3 (安全判定一致性)'
    }
    
    dataset_display = {
        'test1': 'Test1 (校正用)',
        'test2': 'Test2 (獨立測試)',
        'eval': 'Eval (OOD 驗證)'
    }
    
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    for ds in ['test1', 'test2', 'eval']:
        print(f"\n[Dataset: {dataset_display[ds]}]")
        df_ds = df_summary[df_summary['dataset'] == ds]
        
        for tg in ['y1', 'y2', 'y3']:
            print(f"  > Target: {target_display[tg]}")
            df_tg = df_ds[df_ds['target'] == tg]
            
            for mt in ['Brier Score', 'ECE', 'Log Loss']:
                df_mt = df_tg[df_tg['metric'] == mt]
                if not df_mt.empty:
                    row = df_mt.iloc[0]
                    print(f"    |-- {mt:12s}: Best is {row['best_model']} at Layer {row['best_layer']} (Val = {row['best_val']:.4f})")
            print("    L" + "-" * 50)
            
    # Save summary report to CSV for future use
    df_summary.to_csv("results/best_calibrated_models_summary.csv", index=False)
    print("\n[Done] Summary saved to: results/best_calibrated_models_summary.csv")

if __name__ == "__main__":
    main()
