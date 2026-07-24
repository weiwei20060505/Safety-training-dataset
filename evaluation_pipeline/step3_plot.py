import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils_calibration

def plot_reliability_curve(y_true_g, score_pre_g, y_prob_cal_g, bin_edges, title, save_path):
    metrics_raw = utils_calibration.calculate_all_metrics(y_true_g, score_pre_g)
    metrics_cal = utils_calibration.calculate_all_metrics(y_true_g, y_prob_cal_g)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", label="完美校正線 (Perfect Calibration)", alpha=0.5)
    
    # 計算 10 個固定 Bin 區間
    def get_10bin_curve(y_true, y_prob):
        edges = np.linspace(0.0, 1.0, 11)
        bin_ids = np.digitize(y_prob, edges)
        frac_pos = []
        mean_pred = []
        for b in range(1, 11):
            mask = (bin_ids == b)
            if b == 10:
                mask = mask | (y_prob == 1.0)
            n_samples = np.sum(mask)
            if n_samples > 0:
                frac_pos.append(np.mean(y_true[mask]))
                mean_pred.append(np.mean(y_prob[mask]))
            else:
                frac_pos.append(np.nan)
                mean_pred.append((edges[b-1] + edges[b]) / 2.0)
        return np.array(frac_pos), np.array(mean_pred)

    frac_pos_raw, mean_pred_raw = get_10bin_curve(y_true_g, score_pre_g)
    # 過濾 nan
    valid_raw = ~np.isnan(frac_pos_raw)
    ax.plot(mean_pred_raw[valid_raw], frac_pos_raw[valid_raw], "o--", color="#B07A4C", alpha=0.6,
            label=f"校正前 (Raw, Brier: {metrics_raw['brier']:.4f})")
            
    frac_pos_cal, mean_pred_cal = get_10bin_curve(y_true_g, y_prob_cal_g)
    valid_cal = ~np.isnan(frac_pos_cal)
    ax.plot(mean_pred_cal[valid_cal], frac_pos_cal[valid_cal], "s-", color="#4C72B0", linewidth=2.0,
            label=f"校正後 (Isotonic, Brier: {metrics_cal['brier']:.4f})")
            
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel("平均預測機率 (Mean Predicted Probability)", fontsize=11, fontweight='bold')
    ax.set_ylabel("實際正樣本比例 (Fraction of Positives)", fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc="upper left")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_reliability_curve_combined(y_true, score_pre, y_prob_cal, y1_labels, bin_edges, title, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", label="完美校正線 (Perfect Calibration)", alpha=0.5)
    
    # 計算 10 個固定 Bin 區間
    def get_10bin_curve(y_true_sub, y_prob_sub):
        edges = np.linspace(0.0, 1.0, 11)
        bin_ids = np.digitize(y_prob_sub, edges)
        frac_pos = []
        mean_pred = []
        for b in range(1, 11):
            mask = (bin_ids == b)
            if b == 10:
                mask = mask | (y_prob_sub == 1.0)
            n_samples = np.sum(mask)
            if n_samples > 0:
                frac_pos.append(np.mean(y_true_sub[mask]))
                mean_pred.append(np.mean(y_prob_sub[mask]))
            else:
                frac_pos.append(np.nan)
                mean_pred.append((edges[b-1] + edges[b]) / 2.0)
        return np.array(frac_pos), np.array(mean_pred)

    # Group y1 == 0
    mask_0 = (y1_labels == 0)
    if np.sum(mask_0) > 0:
        y_true_0 = y_true[mask_0]
        score_pre_0 = score_pre[mask_0]
        y_prob_cal_0 = y_prob_cal[mask_0]
        
        metrics_raw_0 = utils_calibration.calculate_all_metrics(y_true_0, score_pre_0)
        metrics_cal_0 = utils_calibration.calculate_all_metrics(y_true_0, y_prob_cal_0)
        
        frac_pos_raw_0, mean_pred_raw_0 = get_10bin_curve(y_true_0, score_pre_0)
        valid_raw_0 = ~np.isnan(frac_pos_raw_0)
        ax.plot(mean_pred_raw_0[valid_raw_0], frac_pos_raw_0[valid_raw_0], "o--", color="red", alpha=0.4,
                label=f"y1==0 校正前 (Raw, Brier: {metrics_raw_0['brier']:.4f})")
                
        frac_pos_cal_0, mean_pred_cal_0 = get_10bin_curve(y_true_0, y_prob_cal_0)
        valid_cal_0 = ~np.isnan(frac_pos_cal_0)
        ax.plot(mean_pred_cal_0[valid_cal_0], frac_pos_cal_0[valid_cal_0], "s-", color="red", linewidth=2.0, alpha=0.9,
                label=f"y1==0 校正後 (Isotonic, Brier: {metrics_cal_0['brier']:.4f})")

    # Group y1 == 1
    mask_1 = (y1_labels == 1)
    if np.sum(mask_1) > 0:
        y_true_1 = y_true[mask_1]
        score_pre_1 = score_pre[mask_1]
        y_prob_cal_1 = y_prob_cal[mask_1]
        
        metrics_raw_1 = utils_calibration.calculate_all_metrics(y_true_1, score_pre_1)
        metrics_cal_1 = utils_calibration.calculate_all_metrics(y_true_1, y_prob_cal_1)
        
        frac_pos_raw_1, mean_pred_raw_1 = get_10bin_curve(y_true_1, score_pre_1)
        valid_raw_1 = ~np.isnan(frac_pos_raw_1)
        ax.plot(mean_pred_raw_1[valid_raw_1], frac_pos_raw_1[valid_raw_1], "o--", color="blue", alpha=0.4,
                label=f"y1==1 校正前 (Raw, Brier: {metrics_raw_1['brier']:.4f})")
                
        frac_pos_cal_1, mean_pred_cal_1 = get_10bin_curve(y_true_1, y_prob_cal_1)
        valid_cal_1 = ~np.isnan(frac_pos_cal_1)
        ax.plot(mean_pred_cal_1[valid_cal_1], frac_pos_cal_1[valid_cal_1], "s-", color="blue", linewidth=2.0, alpha=0.9,
                label=f"y1==1 校正後 (Isotonic, Brier: {metrics_cal_1['brier']:.4f})")
                
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel("平均預測機率 (Mean Predicted Probability)", fontsize=11, fontweight='bold')
    ax.set_ylabel("實際正樣本比例 (Fraction of Positives)", fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc="upper left")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_step_mapping(score_pre_g, y_prob_cal_g, title, save_path):
    """
    將分數映射改為長條圖 (Bar Chart) 展示每個原始分數區間映射出的平均機率
    """
    edges = np.linspace(0.0, 1.0, 11)
    bin_labels = [f'{i/10:.1f}-{(i+1)/10:.1f}' for i in range(10)]
    bin_ids = np.digitize(score_pre_g, edges)
    
    mapped_means = []
    for b in range(1, 11):
        mask = (bin_ids == b)
        if b == 10:
            mask = mask | (score_pre_g == 1.0)
        if np.sum(mask) > 0:
            mapped_means.append(np.mean(y_prob_cal_g[mask]))
        else:
            mapped_means.append(0.0)
            
    x_indices = np.arange(10)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x_indices, mapped_means, color="#55A868", edgecolor="black", linewidth=0.7, alpha=0.85, label="保序校正映射機率")
    
    # 在條形圖上方標註具體數值
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
                        
    ax.set_xticks(x_indices)
    ax.set_xticklabels(bin_labels, rotation=45, fontsize=9)
    ax.set_xlim([-0.6, 9.6])
    ax.set_ylim([0, 1.1])
    ax.set_xlabel("原始分數區間 (Raw Score Bins)", fontsize=11, fontweight='bold')
    ax.set_ylabel("校正後機率 (Calibrated Probability)", fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc="upper left")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_brier_components(y_true_g, y_prob_cal_g, title, save_path):
    """
    繪製 2x1 雙 Y 軸 Brier 組分與 Bin 樣本佔比圖表 (精確符合附圖樣式)
    """
    N = len(y_true_g)
    global_mean = np.mean(y_true_g) if N > 0 else 0.0
    
    edges = np.linspace(0.0, 1.0, 11)
    bin_ids = np.digitize(y_prob_cal_g, edges)
    
    rel_vals, res_vals, weight_vals = [], [], []
    for b in range(1, 11):
        mask = (bin_ids == b)
        if b == 10:
            mask = mask | (y_prob_cal_g == 1.0)
        n_samples = np.sum(mask)
        if n_samples > 0:
            w_b = n_samples / N
            pk_bar = np.mean(y_prob_cal_g[mask])
            ok_bar = np.mean(y_true_g[mask])
            rel_b = w_b * ((pk_bar - ok_bar) ** 2)
            res_b = w_b * ((ok_bar - global_mean) ** 2)
        else:
            w_b, rel_b, res_b = 0.0, 0.0, 0.0
        rel_vals.append(rel_b)
        res_vals.append(res_b)
        weight_vals.append(w_b)
        
    x_indices = np.arange(10)
    bin_labels = [f'{i/10:.1f}-{(i+1)/10:.1f}' for i in range(10)]
    bar_width = 0.35
    
    # 建立 2x1 雙視窗 (上方: 可靠度/區分度 雙 Y 軸, 下方: 樣本比例)
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    metrics = utils_calibration.calculate_all_metrics(y_true_g, y_prob_cal_g)
    
    # 上方子圖: 左 Y 軸 (可靠度 Rel - 綠色), 右 Y 軸 (區分度 Res - 紅色)
    colors = {'Reliability': '#55A868', 'Resolution': '#C44E52', 'Weight': '#FFB90F'}
    
    ax_rel = ax_top
    ax_res = ax_top.twinx()
    
    b1 = ax_rel.bar(x_indices - bar_width/2, rel_vals, width=bar_width, color=colors['Reliability'], label='可靠度 (Rel)', zorder=3, alpha=0.85)
    b2 = ax_res.bar(x_indices + bar_width/2, res_vals, width=bar_width, color=colors['Resolution'], label='區分度 (Res)', zorder=3, alpha=0.85)
    
    ax_rel.set_ylabel('Reliability (Rel 貢獻值)', color=colors['Reliability'], fontsize=11, fontweight='bold')
    ax_res.set_ylabel('Resolution (Res 貢獻值)', color=colors['Resolution'], fontsize=11, fontweight='bold')
    ax_rel.tick_params(axis='y', labelcolor=colors['Reliability'])
    ax_res.tick_params(axis='y', labelcolor=colors['Resolution'])
    
    # 合併上方圖例
    lines1, labels1 = ax_rel.get_legend_handles_labels()
    lines2, labels2 = ax_res.get_legend_handles_labels()
    ax_rel.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
    ax_top.grid(True, linestyle='--', alpha=0.3)
    
    # 標題設定
    fig.suptitle(f"{title}\nBrier: {metrics['brier']:.4f} | Rel: {metrics['reliability']:.4f} | Res: {metrics['resolution']:.4f} | Unc: {metrics['uncertainty']:.4f}", 
                 fontsize=12, fontweight='bold', y=0.98)
                 
    # 下方子圖: 樣本比例 Weight
    ax_bottom.bar(x_indices, weight_vals, width=0.45, color=colors['Weight'], edgecolor='black', linewidth=0.7, label='樣本佔比 (Weight)', zorder=3)
    ax_bottom.set_ylabel('Weight 樣本比例', fontsize=11, color='#B8860B', fontweight='bold')
    ax_bottom.set_xlabel('預測分數區間 (Bins)', fontsize=11, fontweight='bold')
    ax_bottom.set_xticks(x_indices)
    ax_bottom.set_xticklabels(bin_labels, rotation=45, fontsize=9)
    max_w = max(weight_vals) if len(weight_vals) > 0 and max(weight_vals) > 0 else 0.5
    ax_bottom.set_ylim([0, min(1.0, max_w * 1.25)])
    ax_bottom.grid(True, linestyle='--', alpha=0.3)
    ax_bottom.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.90)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_quadrant_histograms(pre_scores, post_scores, y1, y2, y3_gt, title, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=15, fontweight='bold')
    
    groups = [
        ((0, 0), "Group 1: Guardrail 放行 (0) | Prompt 真實無害 (0)\n[正常互動]", axes[0, 0]),
        ((0, 1), "Group 2: Guardrail 放行 (0) | Prompt 真實有害 (1)\n[漏報 / 攻擊成功]", axes[0, 1]),
        ((1, 0), "Group 3: Guardrail 攔截 (1) | Prompt 真實無害 (0)\n[誤報 / 過度防禦]", axes[1, 0]),
        ((1, 1), "Group 4: Guardrail 攔截 (1) | Prompt 真實有害 (1)\n[防禦成功]", axes[1, 1]),
    ]
    
    for (val_1, val_2), g_title, ax in groups:
        mask = (y1 == val_1) & (y2 == val_2)
        pre = pre_scores[mask]
        post = post_scores[mask]
        y_true_group = y3_gt[mask]
        n_samples = len(y_true_group)
        
        if n_samples > 0:
            brier, rel, res, unc = utils_calibration.brier_score_decomposition(y_true_group, post, n_bins=10)
            subtitle = f"Brier: {brier:.4f} | Rel(↓): {rel:.4f} | Res(↑): {res:.4f} | Unc: {unc:.4f} (樣本數: {n_samples})"
        else:
            subtitle = "Brier: N/A (無樣本)"
            
        ax.hist(pre, bins=25, range=(0.0, 1.0), color='skyblue', alpha=0.5, label='Pre-cal (Raw)')
        ax.hist(post, bins=25, range=(0.0, 1.0), color='darkorange', alpha=0.6, label='Post-cal (Isotonic)')
        
        ax.set_title(f"{g_title}\n{subtitle}", fontsize=11, fontweight='bold')
        ax.set_xlim([-0.05, 1.05])
        ax.set_xlabel("信心分數 (Score)")
        ax.set_ylabel("次數 (Count)")
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc='upper center')
        
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_score_histograms(pre_scores, post_scores, y_true, title, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    mask_pos = (y_true == 1)
    mask_neg = (y_true == 0)
    n_total = len(y_true) if len(y_true) > 0 else 1.0
    
    # Pre-cal (Raw) - 使用樣本比例 (%) 權重
    ax = axes[0]
    if np.sum(mask_pos) > 0:
        w_pos = np.ones(np.sum(mask_pos)) / n_total * 100.0
        ax.hist(pre_scores[mask_pos], bins=25, range=(0.0, 1.0), weights=w_pos, color='green', alpha=0.5, label=f'真實正樣本 (1) N={np.sum(mask_pos)}')
    if np.sum(mask_neg) > 0:
        w_neg = np.ones(np.sum(mask_neg)) / n_total * 100.0
        ax.hist(pre_scores[mask_neg], bins=25, range=(0.0, 1.0), weights=w_neg, color='red', alpha=0.5, label=f'真實負樣本 (0) N={np.sum(mask_neg)}')
    ax.set_title("校正前 (Raw) 分數比例分布 (%)", fontsize=11, fontweight='bold')
    ax.set_xlim([-0.05, 1.05])
    ax.set_xlabel("信心分數 (Raw Score)", fontsize=10, fontweight='bold')
    ax.set_ylabel("樣本佔比 (%)", fontsize=10, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper center', fontsize=9)
    
    # Post-cal (Calibrated) - 使用樣本比例 (%) 權重
    ax = axes[1]
    if np.sum(mask_pos) > 0:
        w_pos_cal = np.ones(np.sum(mask_pos)) / n_total * 100.0
        ax.hist(post_scores[mask_pos], bins=25, range=(0.0, 1.0), weights=w_pos_cal, color='green', alpha=0.5, label=f'真實正樣本 (1) N={np.sum(mask_pos)}')
    if np.sum(mask_neg) > 0:
        w_neg_cal = np.ones(np.sum(mask_neg)) / n_total * 100.0
        ax.hist(post_scores[mask_neg], bins=25, range=(0.0, 1.0), weights=w_neg_cal, color='red', alpha=0.5, label=f'真實負樣本 (0) N={np.sum(mask_neg)}')
    ax.set_title("校正後 (Isotonic) 分數比例分布 (%)", fontsize=11, fontweight='bold')
    ax.set_xlim([-0.05, 1.05])
    ax.set_xlabel("校正後機率 (Calibrated Probability)", fontsize=10, fontweight='bold')
    ax.set_ylabel("樣本佔比 (%)", fontsize=10, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper center', fontsize=9)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.85)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_joint_calibration(pre_scores, post_scores, y_true, title, save_path):
    fig, ax1 = plt.subplots(figsize=(7, 7))
    
    # 左側 Y 軸: 頻率 Frequency
    ax1.set_xlabel('Confidence', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    
    mask_pos = (y_true == 1)
    mask_neg = (y_true == 0)
    
    # 繪製直方圖 (Correct / Incorrect)
    bins = np.linspace(0.0, 1.0, 21)
    ax1.hist(pre_scores[mask_pos], bins=bins, color='#4CAF50', alpha=0.75, label='Correct', zorder=2)
    ax1.hist(pre_scores[mask_neg], bins=bins, color='#FF4D4D', alpha=0.75, label='Incorrect', zorder=1)
    ax1.tick_params(axis='y')
    
    # 右側 Y 軸: 準確度/機率 Accuracy / Probability
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='blue', fontsize=11, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    # 1. 完美校正對角虛線
    ax2.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    
    # 2. Isotonic regression 階梯映射曲線
    sort_idx = np.argsort(pre_scores)
    x_step = pre_scores[sort_idx]
    y_step = post_scores[sort_idx]
    
    if len(x_step) > 0:
        if x_step[0] > 0.0:
            x_step = np.insert(x_step, 0, 0.0)
            y_step = np.insert(y_step, 0, y_step[0])
        if x_step[-1] < 1.0:
            x_step = np.append(x_step, 1.0)
            y_step = np.append(y_step, y_step[-1])
            
    ax2.plot(x_step, y_step, color='gray', drawstyle='steps-post', linewidth=1.5, label='Isotonic regression')
    
    # 3. Bin accuracy 藍色散點
    bin_centers = []
    bin_accs = []
    for i in range(20):
        low = bins[i]
        high = bins[i+1]
        
        if i == 19:
            bin_mask = (pre_scores >= low) & (pre_scores <= high)
        else:
            bin_mask = (pre_scores >= low) & (pre_scores < high)
            
        if np.sum(bin_mask) > 0:
            acc = np.mean(y_true[bin_mask])
            bin_centers.append((low + high) / 2.0)
            bin_accs.append(acc)
            
    if bin_centers:
        ax2.scatter(bin_centers, bin_accs, facecolors='#7A88FF', edgecolors='blue', s=45, alpha=0.8, label='Bin accuracy', zorder=5)
        
    ax2.set_ylim([-0.05, 1.05])
    
    # 合併雙軸圖例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title(title, fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_metrics_trends(metrics_df, target, split, save_path_brier, save_path_logloss):
    """
    Plots Brier Score and Log Loss trends separately across layers 1-6.
    """
    df_sub = metrics_df[(metrics_df['task'] == target) & (metrics_df['eval_set'] == split)]
    if df_sub.empty:
        return
        
    models = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
    colors = {'SGD': '#4C72B0', 'MLP': '#55A868', 'LGB': '#C44E52', 'LR': '#8172B3', 'RF': '#CCB974'}
    markers = {'SGD': 'o', 'MLP': 's', 'LGB': '^', 'LR': 'v', 'RF': 'D'}
    
    # 1. Plot Brier Score Trend
    fig, ax = plt.subplots(figsize=(8, 6))
    for model in models:
        df_model = df_sub[df_sub['model'] == model].sort_values('layer')
        if df_model.empty:
            continue
        ax.plot(df_model['layer'], df_model['cal_brier'], label=model, color=colors[model],
                marker=markers[model], markersize=6, linewidth=1.5, alpha=0.9)
    ax.set_xticks(range(1, 7))
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlabel('隱藏狀態特徵層數 (Layer)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Brier Score (越低越好)', fontsize=11, fontweight='bold')
    ax.set_title(f'Brier Score 隨層數變化趨勢\n任務: {target} | 評估集: {split}', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    os.makedirs(os.path.dirname(save_path_brier), exist_ok=True)
    fig.savefig(save_path_brier, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # 2. Plot Log Loss Trend
    fig, ax = plt.subplots(figsize=(8, 6))
    for model in models:
        df_model = df_sub[df_sub['model'] == model].sort_values('layer')
        if df_model.empty:
            continue
        ax.plot(df_model['layer'], df_model['cal_logloss'], label=model, color=colors[model],
                marker=markers[model], markersize=6, linewidth=1.5, alpha=0.9)
    ax.set_xticks(range(1, 7))
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlabel('隱藏狀態特徵層數 (Layer)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Log Loss (越低越好)', fontsize=11, fontweight='bold')
    ax.set_title(f'Log Loss 隨層數變化趨勢\n任務: {target} | 評估集: {split}', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    os.makedirs(os.path.dirname(save_path_logloss), exist_ok=True)
    fig.savefig(save_path_logloss, dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_metrics_trends_split_y(predictions_cache, target, split, group_val, save_path_brier, save_path_logloss):
    """
    Plots Brier Score and Log Loss trends separately across layers 1-6 for a specific group (y1 == group_val).
    """
    layers = list(range(1, 7))
    models = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
    colors = {'SGD': '#4C72B0', 'MLP': '#55A868', 'LGB': '#C44E52', 'LR': '#8172B3', 'RF': '#CCB974'}
    markers = {'SGD': 'o', 'MLP': 's', 'LGB': '^', 'LR': 'v', 'RF': 'D'}
    
    records = []
    for layer in layers:
        if target not in predictions_cache or layer not in predictions_cache[target]:
            continue
        if 'splits' not in predictions_cache[target][layer] or split not in predictions_cache[target][layer]['splits']:
            continue
        layer_data = predictions_cache[target][layer]['splits'][split]
        for model in models:
            if model not in layer_data:
                continue
            data = layer_data[model]
            y_true = np.array(data['y_true'])
            score_pre = np.array(data['score_pre'])
            y_prob_cal = np.array(data['y_prob'])
            y1_labels = np.array(data['y1'])
            
            # Mask for the specified group
            mask = (y1_labels == group_val)
            if np.sum(mask) == 0:
                continue
                
            y_true_g = y_true[mask]
            score_pre_g = score_pre[mask]
            y_prob_cal_g = y_prob_cal[mask]
            
            m_raw = utils_calibration.calculate_all_metrics(y_true_g, score_pre_g)
            m_cal = utils_calibration.calculate_all_metrics(y_true_g, y_prob_cal_g)
            
            records.append({
                'model': model,
                'layer': layer,
                'cal_brier': m_cal['brier'],
                'cal_logloss': m_cal['logloss']
            })
            
    df_sub = pd.DataFrame(records)
    if df_sub.empty:
        return
        
    # 1. Plot Brier Score Trend
    fig, ax = plt.subplots(figsize=(8, 6))
    for model in models:
        df_model = df_sub[df_sub['model'] == model].sort_values('layer')
        if df_model.empty:
            continue
        ax.plot(df_model['layer'], df_model['cal_brier'], label=model, color=colors[model],
                marker=markers[model], markersize=6, linewidth=1.5, alpha=0.9)
    ax.set_xticks(range(1, 7))
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlabel('隱藏狀態特徵層數 (Layer)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Brier Score (越低越好)', fontsize=11, fontweight='bold')
    ax.set_title(f'Brier Score 隨層數變化趨勢 (組別: y1 == {group_val})\n任務: {target} | 評估集: {split}', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    os.makedirs(os.path.dirname(save_path_brier), exist_ok=True)
    fig.savefig(save_path_brier, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # 2. Plot Log Loss Trend
    fig, ax = plt.subplots(figsize=(8, 6))
    for model in models:
        df_model = df_sub[df_sub['model'] == model].sort_values('layer')
        if df_model.empty:
            continue
        ax.plot(df_model['layer'], df_model['cal_logloss'], label=model, color=colors[model],
                marker=markers[model], markersize=6, linewidth=1.5, alpha=0.9)
    ax.set_xticks(range(1, 7))
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlabel('隱藏狀態特徵層數 (Layer)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Log Loss (越低越好)', fontsize=11, fontweight='bold')
    ax.set_title(f'Log Loss 隨層數變化趨勢 (組別: y1 == {group_val})\n任務: {target} | 評估集: {split}', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    os.makedirs(os.path.dirname(save_path_logloss), exist_ok=True)
    fig.savefig(save_path_logloss, dpi=150, bbox_inches='tight')
    plt.close(fig)

def main():
    utils_calibration.setup_chinese_font()
    parser = argparse.ArgumentParser(description="LLM Safety Probe Pipeline - Step 3: Granular Plotting Tool")
    parser.add_argument("--target", choices=['y1', 'y2', 'y3', 'all'], default='all', help="Target task to plot")
    parser.add_argument("--split", choices=['test1', 'test2', 'eval', 'all'], default='all', help="Dataset split to plot")
    parser.add_argument("--layer", type=int, choices=[0, 1, 2, 3, 4, 5, 6], default=0, help="Layer num (1-6) to plot, 0 for all")
    parser.add_argument("--model", choices=['SGD', 'MLP', 'LGB', 'LR', 'RF', 'all'], default='all', help="Classifier model to plot")
    parser.add_argument("--chart", choices=['all', 'reliability', 'reliability_combined', 'step_mapping', 'brier_components', 'quadrant_hist', 'score_hist', 'trends', 'trends_split_y', 'joint_calibration'],
                        default='all', help="Chart type to plot")
    args = parser.parse_args()
    
    cache_path = "results/safety_guardrails_evaluation/cache/calibrated_predictions.pkl"
    metrics_path = "results/safety_guardrails_evaluation/cache/all_metrics_records.csv"
    
    if not os.path.exists(cache_path) or not os.path.exists(metrics_path):
        print("錯誤: 找不到預測值快取或指標日誌。請確保已執行 step2_calibrate.py")
        sys.exit(1)
        
    predictions_cache = joblib.load(cache_path)
    metrics_df = pd.read_csv(metrics_path)
    
    targets = ['y1', 'y2', 'y3'] if args.target == 'all' else [args.target]
    splits = ['test1', 'test2', 'eval'] if args.split == 'all' else [args.split]
    layers = list(range(1, 7)) if args.layer == 0 else [args.layer]
    models = ['SGD', 'MLP', 'LGB', 'LR', 'RF'] if args.model == 'all' else [args.model]
    charts = ['reliability', 'reliability_combined', 'step_mapping', 'brier_components', 'quadrant_hist', 'score_hist', 'trends', 'trends_split_y', 'joint_calibration'] if args.chart == 'all' else [args.chart]
    
    bin_edges = np.linspace(0.0, 1.0, 11)
    
    print("開始依據過濾規則繪製圖表...")
    print(f"  篩選條件 -> 任務: {targets} | 評估集: {splits} | 層數: {layers} | 模型: {models} | 圖表類型: {charts}")
    
    # 1. Plot trends
    if 'trends' in charts:
        print("\n[繪製指標隨層數變化趨勢折線圖]...")
        for target in targets:
            for split in splits:
                trend_dir = f"results/plots/01_Metrics_Trends/{target}/{split}"
                brier_path = os.path.join(trend_dir, f"{target}_{split}_brier_score_trend.png")
                logloss_path = os.path.join(trend_dir, f"{target}_{split}_log_loss_trend.png")
                
                plot_metrics_trends(metrics_df, target, split, brier_path, logloss_path)
                print(f"  └─ 產出 {target} {split} Brier & LogLoss 獨立趨勢折線圖")
                
    # 1.1 Plot split trends
    if 'trends_split_y' in charts:
        print("\n[繪製指標隨層數變化趨勢折線圖 (依 y1 分流)]...")
        for target in targets:
            for split in splits:
                for group_val in [0, 1]:
                    trend_split_dir = f"results/plots/01_Metrics_Trends_split_y/{target}/{split}"
                    brier_path = os.path.join(trend_split_dir, f"{target}_{split}_group{group_val}_brier_score_trend.png")
                    logloss_path = os.path.join(trend_split_dir, f"{target}_{split}_group{group_val}_log_loss_trend.png")
                    
                    plot_metrics_trends_split_y(predictions_cache, target, split, group_val, brier_path, logloss_path)
                    print(f"  └─ 產出 {target} {split} 組別 y1=={group_val} Brier & LogLoss 獨立趨勢折線圖")
                    
    # 2. Plot granular model-level charts
    for target in targets:
        for split in splits:
            for layer in layers:
                layer_data = predictions_cache[target][layer]['splits'][split]
                
                for model in models:
                    if model not in layer_data:
                        continue
                    
                    data = layer_data[model]
                    y_true = np.array(data['y_true'])
                    y_prob_cal = np.array(data['y_prob'])
                    score_pre = np.array(data['score_pre'])
                    y1_labels = np.array(data['y1'])
                    y2_labels = np.array(data['y2'])
                    
                    # (A_combined) Reliability Curve (Combined y1)
                    if 'reliability_combined' in charts:
                        rel_comb_dir = f"results/plots/02_Reliability_Curves_combined/{target}/{split}/layer_{layer}"
                        rel_comb_path = os.path.join(rel_comb_dir, f"{target}_{split}_layer{layer}_{model}_combined_reliability.png")
                        rel_comb_title = f"可靠度對比曲線 (合併組別) | 任務: {target} | 評估集: {split} | 層: {layer}\n模型: {model}"
                        plot_reliability_curve_combined(y_true, score_pre, y_prob_cal, y1_labels, bin_edges, rel_comb_title, rel_comb_path)
                    
                    # Reliability, step_mapping, brier_components, score_hist are split by group (y1 == 0 / y1 == 1)
                    for group_val in [0, 1]:
                        group_mask = (y1_labels == group_val)
                        if np.sum(group_mask) == 0:
                            continue
                            
                        y_true_g = y_true[group_mask]
                        y_prob_cal_g = y_prob_cal[group_mask]
                        score_pre_g = score_pre[group_mask]
                        
                        # (A) Reliability Curve (Split y)
                        if 'reliability' in charts:
                            rel_dir = f"results/plots/02_Reliability_Curves_split_y/{target}/{split}/layer_{layer}"
                            rel_path = os.path.join(rel_dir, f"{target}_{split}_layer{layer}_{model}_iso_{group_val}_reliability.png")
                            rel_title = f"可靠度對比曲線 | 任務: {target} | 評估集: {split} | 層: {layer}\n模型: {model} | 分流組別: y1 == {group_val}"
                            plot_reliability_curve(y_true_g, score_pre_g, y_prob_cal_g, bin_edges, rel_title, rel_path)
                            
                        # (B) Step Mapping Bar Chart
                        if 'step_mapping' in charts:
                            step_dir = f"results/plots/06_Step_Mappings/{target}/{split}/layer_{layer}"
                            step_path = os.path.join(step_dir, f"{target}_{split}_layer{layer}_{model}_iso_{group_val}_step_mapping.png")
                            step_title = f"分數映射長條圖 | 任務: {target} | 評估集: {split} | 層: {layer}\n模型: {model} | 分流組別: y1 == {group_val}"
                            plot_step_mapping(score_pre_g, y_prob_cal_g, step_title, step_path)
                            
                        # (C) Brier Components (Dual Subplot Dual y-axis, matching attached sample figure)
                        if 'brier_components' in charts:
                            comp_dir = f"results/plots/05_Brier_Components/{target}/{split}/layer_{layer}"
                            comp_path = os.path.join(comp_dir, f"{target}_{split}_layer{layer}_{model}_iso_{group_val}_brier_components.png")
                            comp_title = f"{model} - Brier 組分與 Bin 樣本佔比 (雙 Y 軸 (Dual Axis))\n{split} | Layer {layer} | {target.upper()} (組別: y1 == {group_val})"
                            plot_brier_components(y_true_g, y_prob_cal_g, comp_title, comp_path)
                            
                        # (D) Score Histograms
                        if 'score_hist' in charts:
                            score_dir = f"results/plots/04_Score_Histograms/{target}/{split}/layer_{layer}"
                            score_path = os.path.join(score_dir, f"{target}_{split}_layer{layer}_{model}_iso_{group_val}_score_histogram.png")
                            score_title = f"正負樣本預測分數直方圖對比\n任務: {target} | 評估集: {split} | 層: {layer} | 模型: {model} | 組別: y1 == {group_val}"
                            y_tgt_g = y1_labels[group_mask] if target == 'y1' else (y2_labels[group_mask] if target == 'y2' else y_true_g)
                            plot_score_histograms(score_pre_g, y_prob_cal_g, y_tgt_g, score_title, score_path)
                            
                        # (F) Joint Calibration Plot
                        if 'joint_calibration' in charts:
                            joint_dir = f"results/plots/07_Joint_Calibration/{target}/{split}/layer_{layer}"
                            joint_path = os.path.join(joint_dir, f"{target}_{split}_layer{layer}_{model}_iso_{group_val}_joint_calibration.png")
                            joint_title = f"Histogram and Scatter Plot of Confidence\nTask: {target.upper()} | Split: {split} | Layer: {layer} | Model: {model} | Group: y1 == {group_val}"
                            plot_joint_calibration(score_pre_g, y_prob_cal_g, y_true_g, joint_title, joint_path)
                            
                    # (E) Quadrant Histograms
                    if 'quadrant_hist' in charts:
                        quad_dir = f"results/plots/03_Quadrant_Histograms/{target}/{split}/layer_{layer}"
                        quad_path = os.path.join(quad_dir, f"{target}_{split}_layer{layer}_{model}_quadrant_histogram.png")
                        quad_title = f"四象限預測置信度直方圖\n任務: {target} | 評估集: {split} | 層: {layer} | 模型: {model}"
                        plot_quadrant_histograms(score_pre, y_prob_cal, y1_labels, y2_labels, y_true, quad_title, quad_path)

    print("\n[OK] 所選圖表繪製流程完成！(圖表皆已儲存至 results/plots/)")

if __name__ == '__main__':
    main()
