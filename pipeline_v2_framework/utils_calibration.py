import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss, log_loss

def setup_chinese_font():
    """Configure matplotlib to use Chinese fonts to prevent rendering issues."""
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'PMingLiU', 'DFKai-SB', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False 

def get_native_bins(y_prob_train):
    """
    Extracts the unique predictions from Isotonic Regression on the training set,
    and returns midpoints as bin edges to define the native steps.
    """
    rounded_prob = np.round(y_prob_train, decimals=5)
    unique_vals = np.unique(rounded_prob)
    if len(unique_vals) <= 1:
        return np.array([-1e-9, 1.0 + 1e-9])
    
    midpoints = [(unique_vals[i] + unique_vals[i+1]) / 2.0 for i in range(len(unique_vals)-1)]
    edges = [-1e-9] + midpoints + [1.0 + 1e-9]
    return np.array(edges)

def get_adaptive_bins(y_prob_train, n_bins=10):
    """
    Groups adjacent isotonic regression step intervals into larger bins without
    splitting any step, ensuring that the 45-degree calibration line on test1 is preserved.
    """
    unique_vals, counts = np.unique(y_prob_train, return_counts=True)
    if len(unique_vals) <= 1:
        return np.array([-1e-9, 1.0 + 1e-9])
        
    N = len(y_prob_train)
    target_bin_size = N / n_bins
    
    bin_edges = []
    curr_count = 0
    for i in range(len(unique_vals) - 1):
        curr_count += counts[i]
        if curr_count >= target_bin_size:
            edge = (unique_vals[i] + unique_vals[i+1]) / 2.0
            bin_edges.append(edge)
            curr_count = 0
            
    edges = [-1e-9] + bin_edges + [1.0 + 1e-9]
    # Deduplicate in case of very few steps
    edges = sorted(list(set(edges)))
    return np.array(edges)

def calculate_calibration_curve(y_true, y_prob, edges):
    """
    Computes actual positive rate and mean prediction in each bin defined by the edges.
    Excludes empty bins to avoid plotting issues.
    """
    clean_prob = np.round(y_prob, decimals=5)
    bin_ids = np.digitize(clean_prob, edges)
    frac_pos = []
    mean_pred = []
    bin_sizes = []
    
    for b in range(1, len(edges)):
        mask = (bin_ids == b)
        n_samples = np.sum(mask)
        if n_samples > 0:
            frac_pos.append(np.mean(y_true[mask]))
            mean_pred.append(np.mean(clean_prob[mask]))
            bin_sizes.append(n_samples)
            
    return np.array(frac_pos), np.array(mean_pred), np.array(bin_sizes)

def calculate_ece(y_true, y_prob, n_bins=10):
    """
    Calculates Expected Calibration Error (ECE) using uniform binning.
    """
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

def brier_score_decomposition(y_true, y_prob, n_bins=10):
    """
    Decomposes the Brier Score into Uncertainty, Reliability, and Resolution.
    Returns: (brier_total, reliability, resolution, uncertainty)
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    N = len(y_true)
    if N == 0:
        return 0.0, 0.0, 0.0, 0.0
        
    # Uncertainty
    o_bar = np.mean(y_true)
    uncertainty = o_bar * (1.0 - o_bar)
    
    # Binning
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(y_prob, bin_edges) - 1
    binids = np.clip(binids, 0, n_bins - 1)
    
    reliability = 0.0
    resolution = 0.0
    
    for k in range(n_bins):
        mask = (binids == k)
        Nk = np.sum(mask)
        if Nk > 0:
            pk_bar = np.nan_to_num(np.mean(y_prob[mask]))
            ok_bar = np.nan_to_num(np.mean(y_true[mask]))
            
            reliability += (Nk / N) * ((pk_bar - ok_bar) ** 2)
            resolution += (Nk / N) * ((ok_bar - o_bar) ** 2)
            
    brier_total = reliability - resolution + uncertainty
    return brier_total, reliability, resolution, uncertainty

def calculate_all_metrics(y_true, y_prob):
    """
    Computes Brier Score, Brier Decomposition, and Log Loss.
    """
    # Clip probabilities to prevent log_loss from blowing up
    y_prob_clipped = np.clip(y_prob, 1e-15, 1 - 1e-15)
    brier_tot, rel, res, unc = brier_score_decomposition(y_true, y_prob)
    try:
        loss = log_loss(y_true, y_prob_clipped)
    except Exception:
        loss = np.nan
    return {
        'brier': brier_tot,
        'uncertainty': unc,
        'reliability': rel,
        'resolution': res,
        'logloss': loss
    }

def plot_comparison_line(models_data, bin_edges_dict, title, save_path):
    """
    Plots calibration line chart for multiple models.
    models_data: dict of model_name -> {'y_true': y_true, 'y_prob': y_prob}
    bin_edges_dict: dict of model_name -> np.array of bin edges
    """
    setup_chinese_font()
    models_list = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
    colors = {'SGD': '#4C72B0', 'MLP': '#55A868', 'LGB': '#C44E52', 'LR': '#8172B3', 'RF': '#CCB974'}
    markers = {'SGD': 'o', 'MLP': 's', 'LGB': '^', 'LR': 'v', 'RF': 'D'}
    
    available_models = [m for m in models_list if m in models_data]
    if not available_models:
        return
        
    plt.figure(figsize=(9, 7))
    plt.plot([0, 1], [0, 1], "k--", label="完美校正線 (理想 45 度)", alpha=0.7)
    
    for model_name in available_models:
        data = models_data[model_name]
        edges = bin_edges_dict[model_name]
        frac_pos, mean_pred, _ = calculate_calibration_curve(data['y_true'], data['y_prob'], edges)
        
        # Calculate metrics for labeling
        metrics = calculate_all_metrics(data['y_true'], data['y_prob'])
        
        plt.plot(mean_pred, frac_pos, marker=markers.get(model_name, 'o'), 
                 color=colors.get(model_name, '#333333'), 
                 label=f"{model_name} (Brier: {metrics['brier']:.4f})", linewidth=1.5)
                 
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("平均預測機率 (S)", fontsize=12)
    plt.ylabel("實際正確比例", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="upper left")
    plt.grid(True, linestyle='--', alpha=0.3)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_side_by_side_bars(models_data, bin_edges_dict, title, save_path):
    """
    Plots side-by-side reliability bar charts for multiple models.
    models_data: dict of model_name -> {'y_true': y_true, 'y_prob': y_prob}
    bin_edges_dict: dict of model_name -> np.array of bin edges
    """
    setup_chinese_font()
    models_list = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
    colors = {'SGD': '#4C72B0', 'MLP': '#55A868', 'LGB': '#C44E52', 'LR': '#8172B3', 'RF': '#CCB974'}
    
    available_models = [m for m in models_list if m in models_data]
    n_models = len(available_models)
    if n_models == 0:
        return
        
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6), sharey=True)
    if n_models == 1:
        axes = [axes]
        
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    for idx, (ax, model_name) in enumerate(zip(axes, available_models)):
        data = models_data[model_name]
        edges = bin_edges_dict[model_name]
        frac_pos, mean_pred, _ = calculate_calibration_curve(data['y_true'], data['y_prob'], edges)
        
        ax.plot([0, 1], [0, 1], "k--", label="完美校正線", zorder=1)
        
        # Plot bars
        n_bins = len(mean_pred)
        width = 0.6 / max(n_bins, 1)
        bars = ax.bar(mean_pred, frac_pos, width=width, color=colors.get(model_name, '#333333'), 
                      alpha=0.85, edgecolor='black', linewidth=0.7, zorder=2)
                      
        # Add labels on top of bars
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, fontweight='bold', color='black')
                            
        metrics = calculate_all_metrics(data['y_true'], data['y_prob'])
        title_str = (f"{model_name}\nBrier: {metrics['brier']:.4f} | Unc: {metrics['uncertainty']:.4f}\n"
                     f"Rel: {metrics['reliability']:.4f} | Res: {metrics['resolution']:.4f}")
        ax.set_title(title_str, fontsize=10, fontweight='bold')
        ax.set_xlabel("平均預測機率 (S)", fontsize=11)
        if idx == 0:
            ax.set_ylabel("實際正確比例", fontsize=11)
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.grid(True, linestyle='--', alpha=0.3)
        
    plt.tight_layout()
    fig.subplots_adjust(top=0.85)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_comparison_line_split_y(models_data, bin_edges_dict, title, save_path):
    """
    Plots a 1x5 grid comparing y1=1 and y1=0 calibration curves for each model.
    models_data: dict of model_name -> {'y_true': y_true, 'y_prob': y_prob, 'y1': y1}
    bin_edges_dict: dict of model_name -> np.array of bin edges
    """
    setup_chinese_font()
    models_list = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
    colors = {'SGD': '#4C72B0', 'MLP': '#55A868', 'LGB': '#C44E52', 'LR': '#8172B3', 'RF': '#CCB974'}
    markers = {'SGD': 'o', 'MLP': 's', 'LGB': '^', 'LR': 'v', 'RF': 'D'}
    
    available_models = [m for m in models_list if m in models_data]
    n_models = len(available_models)
    if n_models == 0:
        return
        
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]
        
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    
    for idx, (ax, model_name) in enumerate(zip(axes, available_models)):
        data = models_data[model_name]
        edges = bin_edges_dict[model_name]
        
        y_true = np.array(data['y_true'])
        y_prob = np.array(data['y_prob'])
        y1 = np.array(data['y1'])
        
        ax.plot([0, 1], [0, 1], "k--", label="完美校正線", alpha=0.5)
        
        # Subsets
        mask_1 = (y1 == 1)
        mask_0 = (y1 == 0)
        
        color = colors.get(model_name, '#333333')
        marker = markers.get(model_name, 'o')
        
        # Plot y1 == 1
        if np.sum(mask_1) > 0:
            frac_pos_1, mean_pred_1, _ = calculate_calibration_curve(y_true[mask_1], y_prob[mask_1], edges)
            m1 = calculate_all_metrics(y_true[mask_1], y_prob[mask_1])
            ax.plot(mean_pred_1, frac_pos_1, marker=marker, color=color, linestyle='-',
                    label=f"y1=1 Unsafe (Brier: {m1['brier']:.4f})", linewidth=1.5)
                    
        # Plot y1 == 0
        if np.sum(mask_0) > 0:
            frac_pos_0, mean_pred_0, _ = calculate_calibration_curve(y_true[mask_0], y_prob[mask_0], edges)
            m0 = calculate_all_metrics(y_true[mask_0], y_prob[mask_0])
            ax.plot(mean_pred_0, frac_pos_0, marker=marker, color=color, linestyle='--',
                    label=f"y1=0 Safe (Brier: {m0['brier']:.4f})", linewidth=1.5, alpha=0.8)
                    
        ax.set_title(f"{model_name}", fontsize=12, fontweight='bold')
        ax.set_xlabel("平均預測機率 (S)", fontsize=11)
        if idx == 0:
            ax.set_ylabel("實際正確比例", fontsize=11)
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_brier_components_bar(models_data, bin_edges_dict, title, save_path):
    """
    繪製分區間 Reliability (Rel) 與 Resolution (Res) 指標柱狀圖
    """
    setup_chinese_font()
    models_list = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
    colors = {'Reliability': '#55A868', 'Resolution': '#C44E52'}
    
    available_models = [m for m in models_list if m in models_data]
    n_models = len(available_models)
    if n_models == 0:
        return
        
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6))
    if n_models == 1:
        axes = [axes]
        
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    
    n_bins = 10
    bin_labels = [f'{i/10:.1f}-{(i+1)/10:.1f}' for i in range(n_bins)]
    x_indices = np.arange(n_bins)
    bar_width = 0.35  # 兩支柱子並列的寬度
    
    for idx, (ax, model_name) in enumerate(zip(axes, available_models)):
        data = models_data[model_name]
        y_true = np.array(data['y_true'])
        y_prob = np.array(data['y_prob'])
        
        global_mean = np.mean(y_true) if len(y_true) > 0 else 0.0
        
        edges = np.linspace(0.0, 1.0, 11)
        bin_ids = np.digitize(y_prob, edges)
        
        rel_vals, res_vals = [], []
        
        for b in range(1, 11):
            mask = (bin_ids == b)
            if b == 10:
                mask = mask | (y_prob == 1.0)
            
            n_samples = np.sum(mask)
            if n_samples > 0:
                y_t_bin = y_true[mask]
                y_p_bin = y_prob[mask]
                
                unc_b = n_samples / len(y_true) if len(y_true) > 0 else 0.0
                rel_b = unc_b * ((np.mean(y_p_bin) - np.mean(y_t_bin)) ** 2)
                res_b = unc_b * ((np.mean(y_t_bin) - global_mean) ** 2)
            else:
                rel_b, res_b = 0.0, 0.0
                
            rel_vals.append(rel_b)
            res_vals.append(res_b)
            
        # 在 ax 畫 Rel 和 Res (不畫 Weight)
        ax.bar(x_indices - bar_width/2, rel_vals, width=bar_width, color=colors['Reliability'], label='可靠度 (Rel)', zorder=3)
        ax.bar(x_indices + bar_width/2, res_vals, width=bar_width, color=colors['Resolution'], label='區分度 (Res)', zorder=3)
        
        # 標題與 X 軸設定
        ax.set_title(f"{model_name}", fontsize=12, fontweight='bold')
        ax.set_xlabel("預測分數區間", fontsize=11)
        ax.set_xticks(x_indices)
        ax.set_xticklabels(bin_labels, rotation=45, fontsize=9)
        
        # Y 軸設定
        ax.set_ylabel("Rel / Res 貢獻值", fontsize=11, color='#333333', fontweight='bold')
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
        
    plt.tight_layout()
    fig.subplots_adjust(top=0.85)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_bin_weights_bar(models_data, bin_edges_dict, title, save_path):
    """
    專門繪製樣本佔比 (Weight) 直方圖
    """
    setup_chinese_font()
    models_list = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
    colors = {'Uncertainty': '#FFB90F'}
    
    available_models = [m for m in models_list if m in models_data]
    n_models = len(available_models)
    if n_models == 0:
        return
        
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6))
    if n_models == 1:
        axes = [axes]
        
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    
    n_bins = 10
    bin_labels = [f'{i/10:.1f}-{(i+1)/10:.1f}' for i in range(n_bins)]
    x_indices = np.arange(n_bins)
    bar_width = 0.5  # 單支柱子寬度
    
    for idx, (ax, model_name) in enumerate(zip(axes, available_models)):
        data = models_data[model_name]
        y_prob = np.array(data['y_prob'])
        
        edges = np.linspace(0.0, 1.0, 11)
        bin_ids = np.digitize(y_prob, edges)
        
        unc_vals = []
        
        for b in range(1, 11):
            mask = (bin_ids == b)
            if b == 10:
                mask = mask | (y_prob == 1.0)
            
            n_samples = np.sum(mask)
            unc_b = n_samples / len(y_prob) if len(y_prob) > 0 else 0.0
            unc_vals.append(unc_b)
            
        ax.bar(x_indices, unc_vals, width=bar_width, color=colors['Uncertainty'], edgecolor='black', linewidth=0.7, label='樣本佔比 (Weight)', zorder=3)
        
        # 標題與 X 軸設定
        ax.set_title(f"{model_name}", fontsize=12, fontweight='bold')
        ax.set_xlabel("預測分數區間", fontsize=11)
        ax.set_xticks(x_indices)
        ax.set_xticklabels(bin_labels, rotation=45, fontsize=9)
        
        # Y 軸設定
        ax.set_ylabel("Weight 樣本比例", fontsize=11, color='#B8860B', fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
        
    plt.tight_layout()
    fig.subplots_adjust(top=0.85)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def compute_dist_stats(y_true, scores):
    """計算依照 y_true (0 vs 1) 分群之預測機率統計量與雙峰分離度指標"""
    from scipy.stats import ks_2samp
    y_true = np.array(y_true)
    scores = np.array(scores)
    
    mask_0 = (y_true == 0)
    mask_1 = (y_true == 1)
    s0 = scores[mask_0]
    s1 = scores[mask_1]
    
    n0, n1 = len(s0), len(s1)
    m0, std0 = (np.mean(s0), np.std(s0)) if n0 > 0 else (0.0, 0.0)
    m1, std1 = (np.mean(s1), np.std(s1)) if n1 > 0 else (0.0, 0.0)
    med0 = np.median(s0) if n0 > 0 else 0.0
    med1 = np.median(s1) if n1 > 0 else 0.0
    q25_0, q75_0 = (np.percentile(s0, 25), np.percentile(s0, 75)) if n0 > 0 else (0.0, 0.0)
    q25_1, q75_1 = (np.percentile(s1, 25), np.percentile(s1, 75)) if n1 > 0 else (0.0, 0.0)
    min0, max0 = (np.min(s0), np.max(s0)) if n0 > 0 else (0.0, 0.0)
    min1, max1 = (np.min(s1), np.max(s1)) if n1 > 0 else (0.0, 0.0)
    
    delta_mu = m1 - m0
    pooled_std = np.sqrt((std0**2 + std1**2) / 2.0) if (std0**2 + std1**2) > 0 else 1e-9
    cohen_d = delta_mu / pooled_std
    
    ks_stat, ks_pval = (ks_2samp(s0, s1).statistic, ks_2samp(s0, s1).pvalue) if (n0 > 0 and n1 > 0) else (0.0, 1.0)
    
    return {
        'n0': n0, 'm0': m0, 'std0': std0, 'med0': med0, 'q25_0': q25_0, 'q75_0': q75_0, 'min0': min0, 'max0': max0,
        'n1': n1, 'm1': m1, 'std1': std1, 'med1': med1, 'q25_1': q25_1, 'q75_1': q75_1, 'min1': min1, 'max1': max1,
        'delta_mu': delta_mu, 'cohen_d': cohen_d, 'ks_stat': ks_stat, 'ks_pval': ks_pval
    }


def print_detailed_bimodal_log(header_title, y_true, score_pre, score_post):
    """列印極詳細之雙峰分佈文字日誌，確保不看圖即可完整獲得統計資訊"""
    stats_pre = compute_dist_stats(y_true, score_pre)
    stats_post = compute_dist_stats(y_true, score_post)
    
    print("\n" + "="*80)
    print(f"【詳細數值日誌】{header_title}")
    print("="*80)
    print(f"  [樣本總數]: N_Total = {len(y_true)} | Class 0 (負例/無害) N0 = {stats_pre['n0']} | Class 1 (正例/有害) N1 = {stats_pre['n1']}")
    print("-"*80)
    
    print("  ► 校正前 (Pre-calibration Raw Scores):")
    print(f"     • Class 0 (Y=0): 平均 = {stats_pre['m0']:.4f} | 標準差 = {stats_pre['std0']:.4f} | 中位數 = {stats_pre['med0']:.4f} | 四分位 [Q1={stats_pre['q25_0']:.4f}, Q3={stats_pre['q75_0']:.4f}] | 範圍 [{stats_pre['min0']:.4f}, {stats_pre['max0']:.4f}]")
    print(f"     • Class 1 (Y=1): 平均 = {stats_pre['m1']:.4f} | 標準差 = {stats_pre['std1']:.4f} | 中位數 = {stats_pre['med1']:.4f} | 四分位 [Q1={stats_pre['q25_1']:.4f}, Q3={stats_pre['q75_1']:.4f}] | 範圍 [{stats_pre['min1']:.4f}, {stats_pre['max1']:.4f}]")
    print(f"     • 雙峰分離度: 平均差 (μ1-μ0) = {stats_pre['delta_mu']:.4f} | Cohen's d = {stats_pre['cohen_d']:.4f} | KS檢定量 = {stats_pre['ks_stat']:.4f} (p-val={stats_pre['ks_pval']:.2e})")
    
    print("-"*80)
    print("  ► 校正後 (Post-calibration Isotonic Scores):")
    print(f"     • Class 0 (Y=0): 平均 = {stats_post['m0']:.4f} | 標準差 = {stats_post['std0']:.4f} | 中位數 = {stats_post['med0']:.4f} | 四分位 [Q1={stats_post['q25_0']:.4f}, Q3={stats_post['q75_0']:.4f}] | 範圍 [{stats_post['min0']:.4f}, {stats_post['max0']:.4f}]")
    print(f"     • Class 1 (Y=1): 平均 = {stats_post['m1']:.4f} | 標準差 = {stats_post['std1']:.4f} | 中位數 = {stats_post['med1']:.4f} | 四分位 [Q1={stats_post['q25_1']:.4f}, Q3={stats_post['q75_1']:.4f}] | 範圍 [{stats_post['min1']:.4f}, {stats_post['max1']:.4f}]")
    print(f"     • 雙峰分離度: 平均差 (μ1-μ0) = {stats_post['delta_mu']:.4f} | Cohen's d = {stats_post['cohen_d']:.4f} | KS檢定量 = {stats_post['ks_stat']:.4f} (p-val={stats_post['ks_pval']:.2e})")
    print("="*80 + "\n")


def plot_bimodal_kde_histogram(y_true, score_pre, score_post, title, save_path):
    """
    繪製預測機率依照真實標籤 (Y_true=0 vs Y_true=1) 分群的 KDE/Histogram 雙峰機率分佈圖。
    繪製 1x2 並列子图 (Left: 校正前 Raw, Right: 校正後 Post-cal)。
    """
    from scipy.stats import gaussian_kde
    setup_chinese_font()
    
    y_true = np.array(y_true)
    score_pre = np.array(score_pre)
    score_post = np.array(score_post)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    x_grid = np.linspace(0.0, 1.0, 200)
    
    plots_config = [
        ("校正前分數雙峰圖 (Raw Score)", score_pre, axes[0]),
        ("校正後分數雙峰圖 (Isotonic)", score_post, axes[1])
    ]
    
    for sub_title, scores, ax in plots_config:
        stats = compute_dist_stats(y_true, scores)
        
        mask_0 = (y_true == 0)
        mask_1 = (y_true == 1)
        s0 = scores[mask_0]
        s1 = scores[mask_1]
        
        # Plot Histograms
        if len(s0) > 0:
            ax.hist(s0, bins=30, range=(0.0, 1.0), density=True, color='#2b5c8f', alpha=0.35, label=f'Y_true=0 (N={len(s0)})')
        if len(s1) > 0:
            ax.hist(s1, bins=30, range=(0.0, 1.0), density=True, color='#d95f02', alpha=0.35, label=f'Y_true=1 (N={len(s1)})')
            
        # Plot KDE Curves
        if len(s0) > 1 and np.std(s0) > 1e-7:
            try:
                kde0 = gaussian_kde(s0, bw_method='scott')
                ax.plot(x_grid, kde0(x_grid), color='#1b3b5f', linewidth=2.5, label='KDE (Y_true=0)')
            except Exception: pass
            
        if len(s1) > 1 and np.std(s1) > 1e-7:
            try:
                kde1 = gaussian_kde(s1, bw_method='scott')
                ax.plot(x_grid, kde1(x_grid), color='#b33e00', linewidth=2.5, label='KDE (Y_true=1)')
            except Exception: pass
            
        # Stats annotation box
        textbox = (f"Y=0: μ={stats['m0']:.3f}, σ={stats['std0']:.3f}\n"
                   f"Y=1: μ={stats['m1']:.3f}, σ={stats['std1']:.3f}\n"
                   f"Δμ: {stats['delta_mu']:.3f} | Cohen's d: {stats['cohen_d']:.3f}\n"
                   f"KS-stat: {stats['ks_stat']:.3f}")
        ax.text(0.5, 0.95, textbox, transform=ax.transAxes, fontsize=10, fontweight='bold',
                verticalalignment='top', horizontalalignment='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='gray'))
                
        ax.set_title(sub_title, fontsize=13, fontweight='bold')
        ax.set_xlabel("預測機率 (Predicted Probability P(Y=1))", fontsize=11)
        ax.set_ylabel("機率密度 (Density)", fontsize=11)
        ax.set_xlim([-0.05, 1.05])
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


