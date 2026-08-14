import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils_calibration import calculate_all_metrics
from step4_combine_plots import combine_grid

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

MODELS = ['MLP', 'LGB', 'LR']
LAYERS = [3, 4, 5, 6]
SPLITS = ['test1', 'test2', 'eval']
TARGET = 'y2'
OUT_DIR = "results/v1_baseline/plots_custom"

colors = {'MLP': '#55A868', 'LGB': '#C44E52', 'LR': '#8172B3'}
markers = {'MLP': 's', 'LGB': '^', 'LR': 'v'}

def custom_plot_metrics_trends(metrics_df, split, out_brier, out_logloss):
    df_sub = metrics_df[(metrics_df['task'] == TARGET) & (metrics_df['eval_set'] == split)]
    df_sub = df_sub[df_sub['layer'].isin(LAYERS)]
    
    # 1. Brier Score
    fig, ax = plt.subplots(figsize=(7, 5))
    for model in MODELS:
        df_model = df_sub[df_sub['model'] == model].sort_values('layer')
        if not df_model.empty:
            ax.plot(df_model['layer'], df_model['cal_brier'], label=model, color=colors[model],
                    marker=markers[model], markersize=7, linewidth=2)
    ax.set_xticks(LAYERS)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel('Layer', fontsize=11, fontweight='bold')
    ax.set_ylabel('Calibrated Brier Score', fontsize=11, fontweight='bold')
    ax.set_title(f'Brier Score Trend - {split.upper()}', fontsize=12, fontweight='bold')
    ax.legend()
    os.makedirs(os.path.dirname(out_brier), exist_ok=True)
    fig.savefig(out_brier, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # 2. Log Loss
    fig, ax = plt.subplots(figsize=(7, 5))
    for model in MODELS:
        df_model = df_sub[df_sub['model'] == model].sort_values('layer')
        if not df_model.empty:
            ax.plot(df_model['layer'], df_model['cal_logloss'], label=model, color=colors[model],
                    marker=markers[model], markersize=7, linewidth=2)
    ax.set_xticks(LAYERS)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel('Layer', fontsize=11, fontweight='bold')
    ax.set_ylabel('Calibrated Log Loss', fontsize=11, fontweight='bold')
    ax.set_title(f'Log Loss Trend - {split.upper()}', fontsize=12, fontweight='bold')
    ax.legend()
    os.makedirs(os.path.dirname(out_logloss), exist_ok=True)
    fig.savefig(out_logloss, dpi=150, bbox_inches='tight')
    plt.close(fig)

def custom_plot_reliability_curve(y_true, score_pre, y_prob_cal, model, layer, split, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.5)
    
    def get_10bin_curve(y_true, y_prob):
        edges = np.linspace(0.0, 1.0, 11)
        bin_ids = np.digitize(y_prob, edges)
        frac_pos, mean_pred = [], []
        for b in range(1, 11):
            mask = (bin_ids == b) | ((b == 10) & (y_prob == 1.0))
            if np.sum(mask) > 0:
                frac_pos.append(np.mean(y_true[mask]))
                mean_pred.append(np.mean(y_prob[mask]))
            else:
                frac_pos.append(np.nan)
                mean_pred.append((edges[b-1] + edges[b]) / 2.0)
        return np.array(frac_pos), np.array(mean_pred)

    f_raw, m_raw = get_10bin_curve(y_true, score_pre)
    v_raw = ~np.isnan(f_raw)
    ax.plot(m_raw[v_raw], f_raw[v_raw], "o--", color="#B07A4C", alpha=0.6, label="Raw")
            
    f_cal, m_cal = get_10bin_curve(y_true, y_prob_cal)
    v_cal = ~np.isnan(f_cal)
    ax.plot(m_cal[v_cal], f_cal[v_cal], "s-", color="#4C72B0", linewidth=2.0, label="Isotonic")
            
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"{model} L{layer} - {split}", fontsize=11, fontweight='bold')
    ax.legend(loc="lower right")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def custom_plot_joint_calibration(score_pre, y_prob_cal, y_true, model, layer, split, save_path):
    fig, ax1 = plt.subplots(figsize=(7.5, 7.5))
    
    ax1.set_xlabel('Confidence', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    
    mask_pos = (y_true == 1)
    mask_neg = (y_true == 0)
    
    bins = np.linspace(0.0, 1.0, 21)
    ax1.hist(score_pre[mask_pos], bins=bins, color='#74c476', alpha=0.75, label='Correct', zorder=2)
    ax1.hist(score_pre[mask_neg], bins=bins, color='#fb6a4a', alpha=0.75, label='Incorrect', zorder=1)
    ax1.tick_params(axis='y', labelcolor='black')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', fontsize=12, color='blue', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    # Perfect calibration
    ax2.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=1.5)
    
    # Isotonic regression line
    sort_idx = np.argsort(score_pre)
    x_step = score_pre[sort_idx]
    y_step = y_prob_cal[sort_idx]
    
    # [FIX] Mask redundant points to prevent rendering artifacts from thousands of overlapping segments
    if len(y_step) > 1:
        mask = np.concatenate(([True], np.diff(y_step) != 0))
        # Ensure the last point is kept so the final horizontal line extends correctly
        mask[-1] = True
        x_step = x_step[mask]
        y_step = y_step[mask]
    
    if len(x_step) > 0:
        if x_step[0] > 0.0:
            x_step = np.insert(x_step, 0, 0.0)
            y_step = np.insert(y_step, 0, y_step[0])
        if x_step[-1] < 1.0:
            x_step = np.append(x_step, 1.0)
            y_step = np.append(y_step, y_step[-1])
            
    ax2.plot(x_step, y_step, color='gray', drawstyle='steps-post', linewidth=1.5, label='Isotonic regression')
    
    # Bin accuracy points
    bin_centers = []
    bin_accs = []
    for i in range(20):
        low = bins[i]
        high = bins[i+1]
        if i == 19:
            bin_mask = (score_pre >= low) & (score_pre <= high)
        else:
            bin_mask = (score_pre >= low) & (score_pre < high)
        if np.sum(bin_mask) > 0:
            bin_centers.append((low + high) / 2.0)
            bin_accs.append(np.mean(y_true[bin_mask]))
            
    ax2.plot(bin_centers, bin_accs, 'bo', markerfacecolor='#a2b9ff', markersize=7, label='Bin accuracy', zorder=5)
    
    # Title
    plt.title(f"Histogram and Scatter Plot of Confidence\nTask: Y2 | Split: {split} | Layer: {layer} | Model: {model}",
              fontweight='bold', fontsize=12)
    
    # Legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', framealpha=0.9, fontsize=9)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def main():
    print("Loading metrics...")
    metrics_csv_path = "cache/v1_baseline/calibration/all_metrics_records.csv"
    if not os.path.exists(metrics_csv_path):
        metrics_csv_path = "cache/v1_baseline/calibration/without_pca/all_metrics_records.csv"
    if not os.path.exists(metrics_csv_path):
        print("Metrics CSV not found!")
        return
    metrics_df = pd.read_csv(metrics_csv_path)

    print("Loading predictions cache...")
    cache_path = "cache/v1_baseline/calibration/calibrated_predictions.pkl"
    if not os.path.exists(cache_path):
        cache_path = "cache/v1_baseline/calibration/without_pca/calibrated_predictions.pkl"
    import joblib
    with open(cache_path, "rb") as f:
        cache = joblib.load(f)

    print("Generating Single Plots...")
    for split in SPLITS:
        # 1. Metrics Trends
        out_brier = f"{OUT_DIR}/01_Metrics_Trends/single/{split}_brier.png"
        out_logloss = f"{OUT_DIR}/01_Metrics_Trends/single/{split}_logloss.png"
        custom_plot_metrics_trends(metrics_df, split, out_brier, out_logloss)
        
        for model in MODELS:
            for layer in LAYERS:
                if TARGET not in cache or layer not in cache[TARGET] or 'splits' not in cache[TARGET][layer] or split not in cache[TARGET][layer]['splits'] or model not in cache[TARGET][layer]['splits'][split]:
                    continue
                d = cache[TARGET][layer]['splits'][split][model]
                y_true = d['y_true']
                score_pre = d['score_pre']
                y_prob = d['y_prob']
                
                # 2. Reliability Curves
                out_rel = f"{OUT_DIR}/02_Reliability_Curves/single/{split}_{model}_L{layer}.png"
                custom_plot_reliability_curve(y_true, score_pre, y_prob, model, layer, split, out_rel)
                
                # 3. Joint Calibration
                out_joint = f"{OUT_DIR}/07_Joint_Calibration/single/{split}_{model}_L{layer}.png"
                custom_plot_joint_calibration(score_pre, y_prob, y_true, model, layer, split, out_joint)

    print("Generating Combined Plots...")
    # 01_Metrics_Trends: 3x2 (rows: splits, cols: logloss, brier)
    grid_01 = []
    for split in SPLITS:
        row = [
            f"{OUT_DIR}/01_Metrics_Trends/single/{split}_logloss.png",
            f"{OUT_DIR}/01_Metrics_Trends/single/{split}_brier.png"
        ]
        grid_01.append(row)
    out_01 = f"{OUT_DIR}/01_Metrics_Trends/combined/trends_3x2.png"
    os.makedirs(os.path.dirname(out_01), exist_ok=True)
    combine_grid(grid_01, out_01, title="Metrics Trends (MLP, LGB, LR)", tile_w=600, tile_h=450,
                 row_labels=[s.upper() for s in SPLITS], col_labels=["Log Loss", "Brier Score"])

    # 02_Reliability_Curves and 07_Joint_Calibration: 3x4 per split (rows: models, cols: layers)
    col_labels = [f"Layer {l}" for l in LAYERS]
    for split in SPLITS:
        # Reliability
        grid_rel = []
        grid_joint = []
        for model in MODELS:
            row_rel = []
            row_joint = []
            for layer in LAYERS:
                row_rel.append(f"{OUT_DIR}/02_Reliability_Curves/single/{split}_{model}_L{layer}.png")
                row_joint.append(f"{OUT_DIR}/07_Joint_Calibration/single/{split}_{model}_L{layer}.png")
            grid_rel.append(row_rel)
            grid_joint.append(row_joint)
        
        out_rel = f"{OUT_DIR}/02_Reliability_Curves/combined/{split}_3x4.png"
        os.makedirs(os.path.dirname(out_rel), exist_ok=True)
        combine_grid(grid_rel, out_rel, title=f"Reliability Curves - {split.upper()}",
                     tile_w=500, tile_h=450, row_labels=MODELS, col_labels=col_labels)

        out_joint = f"{OUT_DIR}/07_Joint_Calibration/combined/{split}_3x4.png"
        os.makedirs(os.path.dirname(out_joint), exist_ok=True)
        combine_grid(grid_joint, out_joint, title=f"Joint Calibration - {split.upper()}",
                     tile_w=500, tile_h=500, row_labels=MODELS, col_labels=col_labels)

    print("All custom plots generated successfully!")

if __name__ == "__main__":
    main()
