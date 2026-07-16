import os
import re
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

# Import preprocessing from project module
from unified_train import DataPreprocessor, DataSplitter

def setup_chinese_font():
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'PMingLiU', 'DFKai-SB', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False 

# ==========================================
# 1. LOG PARSER
# ==========================================
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
            
            m = layer_target_pat.match(line_str)
            if m:
                current_layer = int(m.group(1))
                target_str = m.group(2).strip()
                current_target = target_map.get(target_str, target_str)
                continue
                
            m = model_pat.match(line_str)
            if m:
                current_model = m.group(1).strip()
                continue
                
            m = dataset_pat.match(line_str)
            if m:
                ds_str = m.group(1).strip()
                current_dataset = dataset_map.get(ds_str, ds_str)
                continue
                
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

# ==========================================
# 2. PLOT A: HEATMAP MATRIX (方案 5)
# ==========================================
def generate_advanced_heatmaps(df, output_dir):
    setup_chinese_font()
    models = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
    targets = ['y1', 'y2', 'y3']
    target_display = {
        'y1': 'y1 (模型回覆安全性)',
        'y2': 'y2 (提示詞有害性)',
        'y3': 'y3 (安全判定一致性)'
    }
    metrics = ['Brier Score', 'ECE', 'Log Loss']
    datasets = ['test2', 'eval']
    dataset_titles = {
        'test2': 'Test2 (獨立測試集)',
        'eval': 'Eval (OOD 驗證集)'
    }
    
    for ds in datasets:
        fig, axes = plt.subplots(3, 3, figsize=(16, 14))
        fig.suptitle(f'矯正後模型性能熱圖矩陣 - {dataset_titles[ds]}\n(顏色越深/越藍代表指標越小，性能越好)', 
                     fontsize=18, fontweight='bold', y=0.96)
        
        df_ds = df[df['dataset'] == ds]
        
        for r_idx, tg in enumerate(targets):
            for c_idx, mt in enumerate(metrics):
                ax = axes[r_idx, c_idx]
                
                df_sub = df_ds[(df_ds['target'] == tg) & (df_ds['metric'] == mt)]
                
                matrix = np.zeros((len(models), 6))
                for m_idx, model in enumerate(models):
                    for layer in range(1, 7):
                        df_val = df_sub[(df_sub['model'] == model) & (df_sub['layer'] == layer)]
                        if not df_val.empty:
                            matrix[m_idx, layer - 1] = df_val.iloc[0]['calibrated']
                        else:
                            matrix[m_idx, layer - 1] = np.nan
                            
                im = ax.imshow(matrix, cmap='Blues', aspect='auto', interpolation='nearest')
                
                for i in range(len(models)):
                    for j in range(6):
                        val = matrix[i, j]
                        if not np.isnan(val):
                            color = "white" if val > np.nanmedian(matrix) else "black"
                            ax.text(j, i, f'{val:.4f}', ha="center", va="center", color=color, 
                                    fontsize=9, fontweight='bold')
                
                ax.set_yticks(range(len(models)))
                ax.set_yticklabels(models)
                ax.set_xticks(range(6))
                ax.set_xticklabels([f'L{l}' for l in range(1, 7)])
                
                if r_idx == 0:
                    ax.set_title(f'{mt}', fontsize=13, fontweight='bold')
                if c_idx == 0:
                    ax.set_ylabel(f'{target_display[tg]}', fontsize=12, fontweight='bold')
                if r_idx == 2:
                    ax.set_xlabel('特徵層數 (Layer)', fontsize=11)
                    
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        img_path = os.path.join(output_dir, f'advanced_heatmap_{ds}.png')
        fig.savefig(img_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[進階熱圖] 已儲存至: {img_path}")

# ==========================================
# 3. PLOT B: AVERAGE RANK PLOT (方案 6)
# ==========================================
def generate_rank_plots(df, output_dir):
    setup_chinese_font()
    models = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
    colors_dict = {'SGD': '#4C72B0', 'MLP': '#55A868', 'LGB': '#C44E52', 'LR': '#8172B3', 'RF': '#CCB974'}
    targets = ['y1', 'y2', 'y3']
    target_names = {
        'y1': 'y1 (模型回覆安全性)',
        'y2': 'y2 (提示詞有害性)',
        'y3': 'y3 (安全判定一致性)'
    }
    
    df_rank = df.copy()
    df_rank['rank'] = df_rank.groupby(['dataset', 'target', 'layer', 'metric'])['calibrated'].rank(ascending=True)
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('校正後模型平均排名 (Average Rank) 對比\n(排名越小/條形越短代表越好，1 為最佳)', 
                 fontsize=18, fontweight='bold', y=0.96)
    
    datasets = ['test2', 'eval']
    ds_cols = {'test2': 0, 'eval': 1}
    ds_titles = {'test2': 'Test2 (獨立測試集 - 同分佈)', 'eval': 'Eval (OOD 驗證集 - 跨域)'}
    
    for tg_idx, tg in enumerate(targets):
        for ds in datasets:
            c_idx = ds_cols[ds]
            ax = axes[tg_idx, c_idx]
            
            df_sub = df_rank[(df_rank['dataset'] == ds) & (df_rank['target'] == tg)]
            
            avg_ranks = df_sub.groupby('model')['rank'].mean().reindex(models)
            
            colors = [colors_dict[m] for m in models]
            bars = ax.barh(models, avg_ranks, color=colors, height=0.6, edgecolor='black', linewidth=0.5)
            
            for bar in bars:
                width = bar.get_width()
                ax.annotate(f'{width:.2f}',
                            xy=(width, bar.get_y() + bar.get_height() / 2),
                            xytext=(5, 0), textcoords="offset points",
                            ha='left', va='center', fontsize=10, fontweight='bold')
            
            ax.set_xlim(1.0, 5.2)
            ax.axvline(1.0, color='gold', linestyle='--', alpha=0.7, label='完美排名 (1.0)')
            ax.grid(True, axis='x', linestyle='--', alpha=0.5)
            
            if tg_idx == 0:
                ax.set_title(ds_titles[ds], fontsize=13, fontweight='bold')
            if c_idx == 0:
                ax.set_ylabel(f'{target_names[tg]}\n模型名稱', fontsize=11, fontweight='bold')
            if tg_idx == 2:
                ax.set_xlabel('平均排名 (1-5 越小越好)', fontsize=11)
                
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    img_path = os.path.join(output_dir, 'advanced_model_ranks.png')
    fig.savefig(img_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[排名圖表] 已儲存至: {img_path}")

# ==========================================
# 4. PLOT C: OOD DEGRADATION PLOT (方案 7)
# ==========================================
def generate_degradation_plots(df, output_dir):
    setup_chinese_font()
    models = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
    colors_dict = {'SGD': '#4C72B0', 'MLP': '#55A868', 'LGB': '#C44E52', 'LR': '#8172B3', 'RF': '#CCB974'}
    targets = ['y1', 'y2', 'y3']
    target_names = {
        'y1': 'y1 (模型回覆安全性)',
        'y2': 'y2 (提示詞有害性)',
        'y3': 'y3 (安全判定一致性)'
    }
    metrics = ['Brier Score', 'ECE', 'Log Loss']
    
    df_pivot = df.pivot(index=['target', 'layer', 'metric', 'model'], columns='dataset', values='calibrated').reset_index()
    df_pivot['degradation'] = df_pivot['eval'] - df_pivot['test2']
    
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle('跨域性能退化量 (OOD Degradation: Eval - Test2)\n(數值越小/條形越短代表對抗 OOD 能力越強，負值代表在 OOD 表現反而更好)', 
                 fontsize=18, fontweight='bold', y=0.96)
    
    for r_idx, tg in enumerate(targets):
        for c_idx, mt in enumerate(metrics):
            ax = axes[r_idx, c_idx]
            
            df_sub = df_pivot[(df_pivot['target'] == tg) & (df_pivot['metric'] == mt)]
            
            avg_deg = df_sub.groupby('model')['degradation'].mean().reindex(models)
            
            colors = [colors_dict[m] for m in models]
            bars = ax.barh(models, avg_deg, color=colors, height=0.6, edgecolor='black', linewidth=0.5)
            
            for bar in bars:
                width = bar.get_width()
                ha = 'left' if width >= 0 else 'right'
                offset = 5 if width >= 0 else -5
                ax.annotate(f'{width:+.4f}',
                            xy=(width, bar.get_y() + bar.get_height() / 2),
                            xytext=(offset, 0), textcoords="offset points",
                            ha=ha, va='center', fontsize=9, fontweight='bold')
            
            ax.axvline(0.0, color='black', linestyle='-', linewidth=0.8)
            ax.grid(True, axis='x', linestyle='--', alpha=0.5)
            
            if r_idx == 0:
                ax.set_title(f'{mt}', fontsize=13, fontweight='bold')
            if c_idx == 0:
                ax.set_ylabel(f'{target_names[tg]}', fontsize=12, fontweight='bold')
            if r_idx == 2:
                ax.set_xlabel('平均退化量 (Eval - Test2)', fontsize=11)
                
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    img_path = os.path.join(output_dir, 'advanced_ood_degradation.png')
    fig.savefig(img_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[退化圖表] 已儲存至: {img_path}")

# ==========================================
# 5. QUANTILE ECE CALCULATION
# ==========================================
def calculate_ece_quantile(y_true, y_prob, n_bins=10):
    """
    使用等頻 (Quantile) 分箱計算 Expected Calibration Error (ECE)
    """
    try:
        categories = pd.qcut(pd.Series(y_prob), q=n_bins, labels=False, duplicates='drop')
        ece = 0.0
        n_total = len(y_prob)
        unique_bins = np.unique(categories)
        for b in unique_bins:
            mask = (categories == b)
            n_bin = np.sum(mask)
            if n_bin > 0:
                acc = np.mean(y_true[mask])
                conf = np.mean(y_prob[mask])
                ece += (n_bin / n_total) * np.abs(acc - conf)
        return ece
    except Exception:
        # Fallback to uniform binning if quantile binning fails due to lack of samples
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        binids = np.digitize(y_prob, bin_edges) - 1
        ece = 0.0
        n_total = len(y_prob)
        for i in range(n_bins):
            bin_mask = (binids == i)
            if i == n_bins - 1:
                bin_mask = bin_mask | (y_prob == 1.0)
            n_bin = np.sum(bin_mask)
            if n_bin > 0:
                acc = np.mean(y_true[bin_mask])
                conf = np.mean(y_prob[bin_mask])
                ece += (n_bin / n_total) * np.abs(acc - conf)
        return ece

# ==========================================
# 6. PLOT D: BATCH GENERATION (方案 2 & 3 - 全量產出版)
# ==========================================
def generate_all_before_after_plots(X_3d_train, y_train_dict, X_3d_eval, y_eval_dict, output_dir):
    setup_chinese_font()
    models = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
    colors = {'SGD': '#4C72B0', 'MLP': '#55A868', 'LGB': '#C44E52', 'LR': '#8172B3', 'RF': '#CCB974'}
    markers = {'SGD': 'o', 'MLP': 's', 'LGB': '^', 'LR': 'v', 'RF': 'D'}
    
    target_names = {
        'y1': 'y1 (模型回覆安全性預測)',
        'y2': 'y2 (提示詞有害性預測)',
        'y3': 'y3 (安全判定一致性預測)'
    }
    
    datasets = ['test1', 'test2', 'eval']
    dataset_titles = {
        'test1': 'Test1 (校正用訓練集)',
        'test2': 'Test2 (獨立測試集 - 同分佈)',
        'eval': 'Eval (OOD 外部驗證集)'
    }
    
    num_layers = X_3d_train.shape[1]
    
    # Pre-create output folders for organization
    for ds in datasets:
        os.makedirs(os.path.join(output_dir, ds), exist_ok=True)
        
    print("\n[批次繪圖] 開始生成 54 張機率校正前後對比圖 (Before vs After + 直方圖, Quantile 分箱)...")
    
    for layer_idx in range(1, num_layers + 1):
        for tg_key in ['y1', 'y2', 'y3']:
            y_train = y_train_dict[tg_key]
            y_eval_full = y_eval_dict[tg_key]
            
            # Prepare dataset splits for the layer
            X_2d_train = X_3d_train[:, layer_idx - 1, :]
            X_2d_eval = X_3d_eval[:, layer_idx - 1, :]
            
            _, _, X_test, _, _, y_test, _ = DataSplitter.split_and_scale(X_2d_train, y_train, layer_idx - 1)
            
            # Test1/Test2 split (random_state=42, test_size=0.5)
            X_test1, X_test2, y_test1, y_test2 = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
            
            y_test1_np = np.array(y_test1)
            y_test2_np = np.array(y_test2)
            y_eval_np = np.array(y_eval_full)
            
            # Data containers for plotting
            ds_data = {
                'test1': {'X': X_test1, 'y': y_test1_np},
                'test2': {'X': X_test2, 'y': y_test2_np},
                'eval': {'X': X_2d_eval, 'y': y_eval_np}
            }
            
            # Load models for the current layer and target
            layer_dir = f"results/unified_training/layer_{layer_idx}"
            models_loaded = {}
            for m in models:
                raw_model_path = os.path.join(layer_dir, f"{m.lower()}_{tg_key.lower()}_best.pkl")
                cal_model_path = os.path.join(layer_dir, f"{m.lower()}_{tg_key.lower()}_calibrated.pkl")
                
                if os.path.exists(raw_model_path) and os.path.exists(cal_model_path):
                    models_loaded[m] = {
                        'clf': joblib.load(raw_model_path),
                        'calibrated_clf': joblib.load(cal_model_path)
                    }
                    
            if len(models_loaded) < len(models):
                print(f"  [跳過] 第 {layer_idx} 層 {tg_key} 模型載入不足，無法生成完整圖表。")
                continue
                
            # Loop for each of the 3 datasets
            for ds in datasets:
                X_ds = ds_data[ds]['X']
                y_true = ds_data[ds]['y']
                
                fig, axes = plt.subplots(2, 5, figsize=(18, 8.5))
                
                # Prevent title overlapping: Use a clean suptitle format
                fig.suptitle(f'機率校正前後對比與預測分佈直方圖 - {dataset_titles[ds]}\n第 {layer_idx} 層 - {target_names[tg_key]} (Quantile 等頻分箱)', 
                             fontsize=16, fontweight='bold', y=0.96)
                
                for idx, model in enumerate(models):
                    ax_rel = axes[0, idx]
                    ax_hist = axes[1, idx]
                    
                    # Predict probability of positive class
                    prob_raw = models_loaded[model]['clf'].predict_proba(X_ds)[:, 1]
                    prob_cal = models_loaded[model]['calibrated_clf'].predict_proba(X_ds)[:, 1]
                    
                    # Compute calibration curves using quantile binning
                    frac_raw, mean_raw = calibration_curve(y_true, prob_raw, n_bins=10, strategy='quantile')
                    frac_cal, mean_cal = calibration_curve(y_true, prob_cal, n_bins=10, strategy='quantile')
                    
                    # Calculate ECE using quantile binning
                    ece_raw = calculate_ece_quantile(y_true, prob_raw)
                    ece_cal = calculate_ece_quantile(y_true, prob_cal)
                    
                    # --- Row 1: Reliability curves ---
                    ax_rel.plot([0, 1], [0, 1], "k--", label="完美校正線", alpha=0.7)
                    ax_rel.plot(mean_raw, frac_raw, marker='x', linestyle='--', color='#E66101', 
                                label='Before (校正前)', linewidth=1.2, markersize=5)
                    ax_rel.plot(mean_cal, frac_cal, marker=markers[model], linestyle='-', color=colors[model], 
                                label='After (校正後)', linewidth=1.8, markersize=6)
                    
                    # Subplot Title (use smaller fontsize to avoid overlap)
                    ax_rel.set_title(f'{model}\nECE: {ece_raw:.4f} ➔ {ece_cal:.4f}', fontsize=11, fontweight='bold')
                    ax_rel.set_xlim([-0.05, 1.05])
                    ax_rel.set_ylim([-0.05, 1.05])
                    ax_rel.grid(True, linestyle='--', alpha=0.3)
                    if idx == 0:
                        ax_rel.set_ylabel('實際正確比例\n(Fraction of Positives)', fontsize=11, fontweight='bold')
                    ax_rel.legend(loc='upper left', fontsize=9)
                    
                    # --- Row 2: Confidence Histograms (10 equal bins for distribution density representation) ---
                    # We keep standard density intervals to show forecast clustering, which is standard confidence hist behavior
                    ax_hist.hist(prob_raw, bins=10, range=(0, 1), alpha=0.4, color='#E66101', 
                                 edgecolor='#E66101', label='Before', rwidth=0.85)
                    ax_hist.hist(prob_cal, bins=10, range=(0, 1), alpha=0.5, color=colors[model], 
                                 edgecolor=colors[model], label='After', rwidth=0.85)
                    
                    ax_hist.set_xlabel('預測機率分數 (S)', fontsize=10)
                    if idx == 0:
                        ax_hist.set_ylabel('樣本數量 (Samples)', fontsize=11, fontweight='bold')
                    ax_hist.grid(True, linestyle='--', alpha=0.3)
                    ax_hist.legend(loc='upper right', fontsize=9)
                    ax_hist.set_yscale('log')
                    ax_hist.set_title(f'{model} 分佈 (Log Scale)', fontsize=9, style='italic')
                
                # prevent overlapping between suptitle and subplot titles
                # rect=[0, 0, 1, 0.91] leaves the top 9% of the figure completely free
                # hspace=0.35 and wspace=0.25 separates the panels comfortably
                plt.tight_layout(rect=[0, 0, 1, 0.91])
                fig.subplots_adjust(hspace=0.35, wspace=0.25)
                
                # Save plot
                save_path = os.path.join(output_dir, ds, f'before_after_reliability_{tg_key.lower()}_layer{layer_idx}.png')
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
        print(f"  └─ 第 {layer_idx} / {num_layers} 層所有目標的 3 個資料集對比圖繪製完成")
        
    print(f"[批次繪圖完成] 54 張圖表已成功輸出至: {output_dir}")

# ==========================================
# 7. PLOT E: BRIER SCORE DECOMPOSITION (方案 4)
# ==========================================
def calculate_brier_decomposition(y_true, y_prob, n_bins=10):
    N = len(y_prob)
    if N == 0:
        return 0, 0, 0
    o_bar = np.mean(y_true)
    
    # Discretize using quantile boundaries to be consistent
    try:
        categories = pd.qcut(pd.Series(y_prob), q=n_bins, labels=False, duplicates='drop')
        unique_bins = np.unique(categories)
        rel = 0.0
        res = 0.0
        for t in unique_bins:
            mask = (categories == t)
            N_t = np.sum(mask)
            if N_t > 0:
                f_t = np.mean(y_prob[mask])
                o_t = np.mean(y_true[mask])
                rel += N_t * ((f_t - o_t) ** 2)
                res += N_t * ((o_t - o_bar) ** 2)
        rel = rel / N
        res = res / N
    except Exception:
        # Fallback to uniform
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        binids = np.digitize(y_prob, bin_edges) - 1
        rel = 0.0
        res = 0.0
        for t in range(n_bins):
            bin_mask = (binids == t)
            if t == n_bins - 1:
                bin_mask = bin_mask | (y_prob == 1.0)
            N_t = np.sum(bin_mask)
            if N_t > 0:
                f_t = np.mean(y_prob[bin_mask])
                o_t = np.mean(y_true[bin_mask])
                rel += N_t * ((f_t - o_t) ** 2)
                res += N_t * ((o_t - o_bar) ** 2)
        rel = rel / N
        res = res / N
        
    unc = o_bar * (1.0 - o_bar)
    return rel, res, unc

def generate_brier_decomposition(pred_data, output_dir):
    setup_chinese_font()
    models = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
    
    decomp_data = []
    
    for model in models:
        if model not in pred_data:
            continue
        data = pred_data[model]
        y_true = data['y_true']
        
        rel_raw, res_raw, unc_raw = calculate_brier_decomposition(y_true, data['prob_raw'])
        bs_raw = brier_score_loss(y_true, data['prob_raw'])
        
        rel_cal, res_cal, unc_cal = calculate_brier_decomposition(y_true, data['prob_cal'])
        bs_cal = brier_score_loss(y_true, data['prob_cal'])
        
        decomp_data.append({
            'Model': model,
            'Type': 'Before (校正前)',
            'Reliability': rel_raw,
            'Resolution': res_raw,
            'Uncertainty': unc_raw,
            'Brier Score': bs_raw
        })
        decomp_data.append({
            'Model': model,
            'Type': 'After (校正後)',
            'Reliability': rel_cal,
            'Resolution': res_cal,
            'Uncertainty': unc_cal,
            'Brier Score': bs_cal
        })
        
    df_decomp = pd.DataFrame(decomp_data)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(models))
    width = 0.18
    
    df_before = df_decomp[df_decomp['Type'] == 'Before (校正前)'].set_index('Model').reindex(models)
    df_after = df_decomp[df_decomp['Type'] == 'After (校正後)'].set_index('Model').reindex(models)
    
    unc_val = df_decomp['Uncertainty'].iloc[0] if not df_decomp.empty else 0.0
    
    rects1 = ax.bar(x - width*1.5, df_before['Brier Score'], width, label='Brier Score (Before)', color='#FDB863', edgecolor='black', linewidth=0.5)
    rects2 = ax.bar(x - width*0.5, df_after['Brier Score'], width, label='Brier Score (After)', color='#B2ABD2', edgecolor='black', linewidth=0.5)
    
    rects3 = ax.bar(x + width*0.5, df_before['Reliability'], width, label='Reliability (Before, 越低越好)', color='#E66101', edgecolor='black', linewidth=0.5, hatch='//')
    rects4 = ax.bar(x + width*1.5, df_after['Reliability'], width, label='Reliability (After, 越低越好)', color='#5E3C99', edgecolor='black', linewidth=0.5, hatch='//')
    
    ax.scatter(x - width, df_before['Resolution'], color='#228B22', marker='D', s=50, zorder=4, label='Resolution (Before, 越高越好)')
    ax.scatter(x + width, df_after['Resolution'], color='#32CD32', marker='D', s=50, zorder=4, label='Resolution (After, 越高越好)')
    
    ax.axhline(unc_val, color='red', linestyle='--', linewidth=1.5, label=f'Uncertainty (資料集不確定度 = {unc_val:.4f})')
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
                        
    autolabel(rects1)
    autolabel(rects2)
    
    ax.set_title('Brier Score 三因子分解與對比 (Test2 獨立測試集)\nBrier Score (BS) = Reliability - Resolution + Uncertainty', 
                 fontsize=15, fontweight='bold')
    ax.set_ylabel('指標數值 (Value)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.set_ylim([0, max(df_decomp['Brier Score'].max() * 1.25, unc_val * 1.25)])
    
    ax.legend(loc='upper right', ncol=2, fontsize=9)
    
    plt.tight_layout()
    img_path = os.path.join(output_dir, 'advanced_brier_decomposition.png')
    fig.savefig(img_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Brier 分解圖] 已儲存至: {img_path}")

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    log_filepath = "results/reliability_diagrams/calibration_metrics_log.txt"
    output_dir = "results/reliability_diagrams/advanced"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print(" 啟動進階機率校正多維度指標分析與批量繪圖工具")
    print("=" * 60)
    
    # 1. Parse log file
    df = parse_log_file(log_filepath)
    if df is None or df.empty:
        print("錯誤: 無法解析指標日誌檔！")
        sys.exit(1)
    print(f"成功解析共 {len(df)} 筆指標數據。")
    
    # 2. Plot A: Heatmaps (Scheme 5)
    generate_advanced_heatmaps(df, output_dir)
    
    # 3. Plot B: Ranks (Scheme 6)
    generate_rank_plots(df, output_dir)
    
    # 4. Plot C: OOD Degradation (Scheme 7)
    generate_degradation_plots(df, output_dir)
    
    # 5. Batch load and generate Plot D (Scheme 2 & 3 for all layers, datasets and targets)
    TRAIN_PATH = "experiment_results_train_10000.pkl"
    EVAL_PATH = "experiment_results_eval.pkl"
    
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(EVAL_PATH):
        print(f"錯誤: 找不到訓練或評估檔案 {TRAIN_PATH} 或 {EVAL_PATH}")
        sys.exit(1)
        
    print(f"\n[1] 載入訓練數據集: {TRAIN_PATH}...")
    prep_train = DataPreprocessor(TRAIN_PATH)
    prep_train.load_data()
    X_3d_train = prep_train.extract_features()
    y1_train, y2_train, y3_train = prep_train.create_targets()
    y_train_dict = {'y1': y1_train, 'y2': y2_train, 'y3': y3_train}
    
    print(f"[2] 載入外部數據集: {EVAL_PATH}...")
    prep_eval = DataPreprocessor(EVAL_PATH)
    prep_eval.load_data()
    X_3d_eval = prep_eval.extract_features()
    y1_eval, y2_eval, y3_eval = prep_eval.create_targets()
    y_eval_dict = {'y1': y1_eval, 'y2': y2_eval, 'y3': y3_eval}
    
    # Run the 54-plot generation loop
    generate_all_before_after_plots(X_3d_train, y_train_dict, X_3d_eval, y_eval_dict, output_dir)
    
    # 6. Generate Plot E: Brier Score Decomposition for a representative case
    # (using Target: y1, Layer: 6 on test2 to keep it comparable)
    print("\n[Brier 分解] 生成代表性 Brier 分解圖 (y1, Layer 6)...")
    try:
        # Load predictions for y1, Layer 6
        layer_6 = 6
        target_y1 = 'y1'
        
        # Prepare datasets split
        X_2d_train_6 = X_3d_train[:, layer_6 - 1, :]
        _, _, X_test_6, _, _, y_test_6, _ = DataSplitter.split_and_scale(X_2d_train_6, y1_train, layer_6 - 1)
        _, X_test2_6, _, y_test2_6 = train_test_split(X_test_6, y_test_6, test_size=0.5, random_state=42)
        y_test2_6_np = np.array(y_test2_6)
        
        layer_dir_6 = f"results/unified_training/layer_{layer_6}"
        pred_data_6 = {}
        for m in ['SGD', 'MLP', 'LGB', 'LR', 'RF']:
            raw_path = os.path.join(layer_dir_6, f"{m.lower()}_{target_y1.lower()}_best.pkl")
            cal_path = os.path.join(layer_dir_6, f"{m.lower()}_{target_y1.lower()}_calibrated.pkl")
            if os.path.exists(raw_path) and os.path.exists(cal_path):
                clf_raw = joblib.load(raw_path)
                clf_cal = joblib.load(cal_path)
                pred_data_6[m] = {
                    'y_true': y_test2_6_np,
                    'prob_raw': clf_raw.predict_proba(X_test2_6)[:, 1],
                    'prob_cal': clf_cal.predict_proba(X_test2_6)[:, 1]
                }
        
        generate_brier_decomposition(pred_data_6, output_dir)
    except Exception as e:
        print(f"[錯誤] Brier 分解圖產出失敗: {str(e)}")
        
    print("\n" + "=" * 60)
    print(" 所有進階分析與批量對比圖（共 54 張及熱圖、排名圖等）繪製已全部完成！")
    print(" 產出目錄: results/reliability_diagrams/advanced/")
    print("=" * 60)

if __name__ == "__main__":
    main()
