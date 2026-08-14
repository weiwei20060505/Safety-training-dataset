"""
8月6日 LLM 隱藏狀態機率校正與元評估框架 — 階段二核心可視化圖表 (V2)
======================================================================
針對 Layer 6 繪製高清圖片：
1. 07_Joint_Calibration：Histogram and Scatter Plot of Confidence 雙 Y 軸聯合校正圖
2. 01_Metrics_Trends_split_y：Brier Score 與 Log Loss 隨層數變化趨勢圖 (目前僅 1 層)
3. 02_Reliability_Curves_combined：6 個模型獨自繪製校正前後 y1=0 / y1=1 組合大圖
"""

import os
import sys
# Add current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from utils_calibration import calculate_all_metrics

# 字態與風格設定
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

ALL_LAYERS = [6]
ALL_SPLITS = ['test1', 'test2', 'eval']
ALL_MODELS = [
    'LR_Hard_Dual',
    'LR_Interaction',
    'LGB_Hard_Dual',
    'RootSplit_LGBM',
    'MLP_Hard_Dual',
    'YHead_MLP'
]
MODEL_DISPLAY_NAMES = {
    'LR_Hard_Dual': 'LR-HardDual',
    'LR_Interaction': 'LR-Interaction',
    'LGB_Hard_Dual': 'LGB-HardDual',
    'RootSplit_LGBM': 'LGB-RootSplit',
    'MLP_Hard_Dual': 'MLP-HardDual',
    'YHead_MLP': 'MLP-YHead'
}
MODEL_COLORS = {
    'LR_Hard_Dual': '#1f77b4',       # Blue
    'LR_Interaction': '#ff7f0e',     # Orange
    'LGB_Hard_Dual': '#2ca02c',      # Green
    'RootSplit_LGBM': '#d62728',     # Red
    'MLP_Hard_Dual': '#9467bd',      # Purple
    'YHead_MLP': '#8c564b'           # Brown
}
MODEL_MARKERS = {
    'LR_Hard_Dual': 'o',
    'LR_Interaction': 'v',
    'LGB_Hard_Dual': 's',
    'RootSplit_LGBM': 'p',
    'MLP_Hard_Dual': '*',
    'YHead_MLP': 'D'
}

# ─── PIL 拼圖輔助函數 ──────────────────────────────────────────────────────────

def _load_font(size: int = 18) -> ImageFont.FreeTypeFont:
    candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\Arial.ttf",
        r"C:\Windows\Fonts\msjh.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()

def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple:
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        return draw.textsize(text, font=font)

def combine_grid(image_paths_2d, output_path, title="", tile_w=700, tile_h=550, row_labels=None, col_labels=None):
    n_rows = len(image_paths_2d)
    n_cols = max((len(r) for r in image_paths_2d), default=0)
    if n_rows == 0 or n_cols == 0:
        return

    TITLE_H = 60 if title else 0
    COL_LBL_H = 40 if col_labels else 0
    ROW_LBL_W = 100 if row_labels else 0

    canvas_w = ROW_LBL_W + n_cols * tile_w
    canvas_h = TITLE_H + COL_LBL_H + n_rows * tile_h

    canvas = Image.new('RGB', (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    font_title = _load_font(24)
    font_label = _load_font(18)

    if title:
        tw, th = _text_size(draw, title, font_title)
        draw.text(((canvas_w - tw) // 2, (TITLE_H - th) // 2), title, fill=(20, 20, 20), font=font_title)

    if col_labels:
        for c, lbl in enumerate(col_labels[:n_cols]):
            cx = ROW_LBL_W + c * tile_w + tile_w // 2
            cy = TITLE_H + COL_LBL_H // 2
            tw, th = _text_size(draw, lbl, font_label)
            draw.text((cx - tw // 2, cy - th // 2), lbl, fill=(40, 40, 160), font=font_label)

    for r, row in enumerate(image_paths_2d):
        row_y = TITLE_H + COL_LBL_H + r * tile_h
        if row_labels and r < len(row_labels):
            tw, th = _text_size(draw, row_labels[r], font_label)
            draw.text((ROW_LBL_W // 2 - tw // 2, row_y + tile_h // 2 - th // 2), row_labels[r], fill=(160, 40, 40), font=font_label)

        for c in range(n_cols):
            img_path = row[c] if c < len(row) else None
            cell_x = ROW_LBL_W + c * tile_w
            cell_y = row_y

            if img_path and os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((tile_w, tile_h), Image.LANCZOS)
                    canvas.paste(img, (cell_x, cell_y))
                except Exception:
                    pass

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas.save(output_path)
    print(f"  └─ 已成功儲存組合圖: {output_path}")


# ─── 07_Joint_Calibration (與附圖完全一致) ────────────────────────────────────

def plot_single_joint_calibration(pre_scores, post_scores, y_true, title, save_path):
    """
    繪製單張 Histogram and Scatter Plot of Confidence
    雙 Y 軸 (左: Frequency, 右: Accuracy)
    """
    fig, ax1 = plt.subplots(figsize=(7.5, 7.5))

    ax1.set_xlabel('Confidence', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')

    mask_pos = (y_true == 1)
    mask_neg = (y_true == 0)

    bins = np.linspace(0.0, 1.0, 21)
    ax1.hist(pre_scores[mask_pos], bins=bins, color='#74c476', alpha=0.75, label='Correct', zorder=2)
    ax1.hist(pre_scores[mask_neg], bins=bins, color='#fb6a4a', alpha=0.75, label='Incorrect', zorder=1)
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='blue', fontsize=11, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='blue')

    # 1. 完美校正線
    ax2.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=1.5)

    # 2. Isotonic regression 階梯線
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

    # 3. Bin accuracy 藍點
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

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

    plt.title(title, fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)

def generate_07_joint_calibration(calib_data, base_output_dir, pca_status):
    print("\n" + "="*80)
    print(f"【圖表 1/3】生成 07_Joint_Calibration (Confidence 聯合校正圖) | PCA: {pca_status}")
    print("="*80)

    single_dir = os.path.join(base_output_dir, "07_Joint_Calibration", "single", pca_status)
    combined_dir = os.path.join(base_output_dir, "07_Joint_Calibration", "combined", pca_status)

    for split in ALL_SPLITS:
        for layer in ALL_LAYERS:
            for model_name in ALL_MODELS:
                try:
                    info = calib_data['layers'][layer][model_name][split]
                except KeyError:
                    continue

                score_pre = info['score_pre']
                p_cal = info['prob_cal_split']
                y1 = info['y1']
                y3 = info['y3']

                # 1. Group y1 == 0
                mask0 = (y1 == 0)
                if np.sum(mask0) > 0:
                    title0 = f"Histogram and Scatter Plot of Confidence\nTask: Y3 | Split: {split} | Layer: {layer} | Model: {MODEL_DISPLAY_NAMES[model_name]} | Group: y1 == 0"
                    save0 = os.path.join(single_dir, split, f"joint_cal_layer{layer}_{model_name}_group0.png")
                    plot_single_joint_calibration(score_pre[mask0], p_cal[mask0], y3[mask0], title0, save0)

                # 2. Group y1 == 1
                mask1 = (y1 == 1)
                if np.sum(mask1) > 0:
                    title1 = f"Histogram and Scatter Plot of Confidence\nTask: Y3 | Split: {split} | Layer: {layer} | Model: {MODEL_DISPLAY_NAMES[model_name]} | Group: y1 == 1"
                    save1 = os.path.join(single_dir, split, f"joint_cal_layer{layer}_{model_name}_group1.png")
                    plot_single_joint_calibration(score_pre[mask1], p_cal[mask1], y3[mask1], title1, save1)

    print("  └─ 開始按 Group 生成 3×6 組合大圖...")
    col_labels = [MODEL_DISPLAY_NAMES[m] for m in ALL_MODELS]
    row_labels = [s.upper() for s in ALL_SPLITS]

    layer = ALL_LAYERS[0] # assuming layer 6 only

    for g in [0, 1]:
        grid = []
        for split in ALL_SPLITS:
            row = []
            for model_name in ALL_MODELS:
                p = os.path.join(single_dir, split, f"joint_cal_layer{layer}_{model_name}_group{g}.png")
                row.append(p)
            grid.append(row)

        out_comb = os.path.join(combined_dir, f"joint_cal_combined_3x6_group{g}.png")
        combine_grid(
            grid, out_comb,
            title=f"Joint Calibration 3x6 Overview (Group: y1 == {g}) | ({pca_status})",
            tile_w=650, tile_h=650,
            row_labels=row_labels,
            col_labels=col_labels
        )

    print("  └─ 開始按模型生成 2×3 組合大圖 (列: y1分群, 欄: 測試集)...")
    col_labels_2x3 = [s.upper() for s in ALL_SPLITS]
    row_labels_2x3 = ["y1 == 0", "y1 == 1"]

    for model_name in ALL_MODELS:
        grid_2x3 = []
        for g in [0, 1]:
            row = []
            for split in ALL_SPLITS:
                p = os.path.join(single_dir, split, f"joint_cal_layer{layer}_{model_name}_group{g}.png")
                row.append(p)
            grid_2x3.append(row)
        
        out_comb_2x3 = os.path.join(combined_dir, f"joint_cal_combined_2x3_{model_name}.png")
        combine_grid(
            grid_2x3, out_comb_2x3,
            title=f"Joint Calibration Across Splits (y1=0 vs y1=1) — Model: {MODEL_DISPLAY_NAMES[model_name]} | ({pca_status})",
            tile_w=650, tile_h=650,
            row_labels=row_labels_2x3,
            col_labels=col_labels_2x3
        )

    print("  └─ 開始按測試集生成 2×6 組合大圖 (列: y1分群, 欄: 6種模型)...")
    col_labels_2x6 = [MODEL_DISPLAY_NAMES[m] for m in ALL_MODELS]
    row_labels_2x6 = ["y1 == 0", "y1 == 1"]

    for split in ALL_SPLITS:
        grid_2x6 = []
        for g in [0, 1]:
            row = []
            for model_name in ALL_MODELS:
                p = os.path.join(single_dir, split, f"joint_cal_layer{layer}_{model_name}_group{g}.png")
                row.append(p)
            grid_2x6.append(row)
        
        out_comb_2x6 = os.path.join(combined_dir, f"joint_cal_combined_2x6_{split}.png")
        combine_grid(
            grid_2x6, out_comb_2x6,
            title=f"Joint Calibration Across Models (y1=0 vs y1=1) — Split: {split.upper()} | ({pca_status})",
            tile_w=650, tile_h=650,
            row_labels=row_labels_2x6,
            col_labels=col_labels_2x6
        )


# ─── 01_Metrics_Trends_split_y (Y 軸全域強制統一) ────────────────────────────

def generate_01_metrics_trends_split_y(calib_data, base_output_dir, pca_status):
    print("\n" + "="*80)
    print(f"【圖表 2/3】生成 01_Metrics_Trends_split_y (Brier & LogLoss 趨勢圖 - 統一 Y 軸) | PCA: {pca_status}")
    print("="*80)

    single_dir = os.path.join(base_output_dir, "01_Metrics_Trends_split_y", "single", pca_status)
    combined_dir = os.path.join(base_output_dir, "01_Metrics_Trends_split_y", "combined", pca_status)

    # 先收集所有數據以計算全域統一 Y 軸範圍
    all_records = []
    for layer in ALL_LAYERS:
        for model_name in ALL_MODELS:
            for split in ALL_SPLITS:
                try:
                    info = calib_data['layers'][layer][model_name][split]
                except KeyError:
                    continue

                y1 = info['y1']
                y3 = info['y3']
                p_cal = info['prob_cal_split']

                for g in [0, 1]:
                    mask = (y1 == g)
                    if np.sum(mask) == 0:
                        continue
                    m_cal = calculate_all_metrics(y3[mask], p_cal[mask])
                    all_records.append({
                        'layer': layer,
                        'model': model_name,
                        'split': split,
                        'group': g,
                        'brier': m_cal['brier'],
                        'logloss': m_cal['logloss']
                    })

    df = pd.DataFrame(all_records)
    if df.empty:
        return

    # 全域統一 Y 軸範圍
    brier_min, brier_max = df['brier'].min(), df['brier'].max()
    brier_ylim = (max(0.0, brier_min * 0.9), brier_max * 1.1)

    logloss_valid = df['logloss'].dropna()
    logloss_min, logloss_max = logloss_valid.min(), logloss_valid.max()
    logloss_ylim = (max(0.0, logloss_min * 0.9), logloss_max * 1.1)

    print(f"  ├─ [統一 Y 軸刻度] Brier Range: [{brier_ylim[0]:.4f}, {brier_ylim[1]:.4f}]")
    print(f"  └─ [統一 Y 軸刻度] LogLoss Range: [{logloss_ylim[0]:.4f}, {logloss_ylim[1]:.4f}]")

    # 繪製單圖
    for split in ALL_SPLITS:
        split_single_dir = os.path.join(single_dir, split)
        for g in [0, 1]:
            df_sg = df[(df['split'] == split) & (df['group'] == g)]

            # 1. Brier Bar Chart
            fig, ax = plt.subplots(figsize=(8.5, 5.5))
            x_pos = np.arange(len(ALL_MODELS))
            brier_vals = []
            for model_name in ALL_MODELS:
                df_m = df_sg[df_sg['model'] == model_name]
                brier_vals.append(df_m['brier'].values[0] if not df_m.empty else 0)
            
            bars = ax.bar(x_pos, brier_vals, color=[MODEL_COLORS[m] for m in ALL_MODELS], alpha=0.85, edgecolor='black', width=0.6)
            ax.set_xticks(x_pos)
            ax.set_xticklabels([MODEL_DISPLAY_NAMES[m] for m in ALL_MODELS], rotation=15, fontsize=9)
            ax.set_ylim(brier_ylim)
            ax.set_ylabel('Brier Score (Lower is Better)', fontsize=11, fontweight='bold')
            ax.set_title(f'Brier Score Comparison (Group: y1 == {g}) | Split: {split.upper()}', fontsize=12, fontweight='bold')
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.annotate(f'{h:.4f}', xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0,3), textcoords='offset points', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            save_brier = os.path.join(split_single_dir, f"brier_trend_group{g}.png")
            os.makedirs(os.path.dirname(save_brier), exist_ok=True)
            fig.savefig(save_brier, dpi=180, bbox_inches='tight')
            plt.close(fig)

            # 2. Log Loss Bar Chart
            fig, ax = plt.subplots(figsize=(8.5, 5.5))
            logloss_vals = []
            for model_name in ALL_MODELS:
                df_m = df_sg[df_sg['model'] == model_name]
                logloss_vals.append(df_m['logloss'].values[0] if not df_m.empty else 0)
                
            bars = ax.bar(x_pos, logloss_vals, color=[MODEL_COLORS[m] for m in ALL_MODELS], alpha=0.85, edgecolor='black', width=0.6)
            ax.set_xticks(x_pos)
            ax.set_xticklabels([MODEL_DISPLAY_NAMES[m] for m in ALL_MODELS], rotation=15, fontsize=9)
            ax.set_ylim(logloss_ylim)
            ax.set_ylabel('Log Loss (Lower is Better)', fontsize=11, fontweight='bold')
            ax.set_title(f'Log Loss Comparison (Group: y1 == {g}) | Split: {split.upper()}', fontsize=12, fontweight='bold')
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.annotate(f'{h:.4f}', xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0,3), textcoords='offset points', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            save_logloss = os.path.join(split_single_dir, f"logloss_trend_group{g}.png")
            os.makedirs(os.path.dirname(save_logloss), exist_ok=True)
            fig.savefig(save_logloss, dpi=180, bbox_inches='tight')
            plt.close(fig)

    print("  └─ 開始按 Split 生成 1×2 組合大圖，並保存 3x2 全域總圖...")
    
    # 1. 產出各測試集 (split) 自有的 1x2 組合圖
    col_labels_1x2 = ["Group: y1 == 0", "Group: y1 == 1"]
    for split in ALL_SPLITS:
        split_single_dir = os.path.join(single_dir, split)
        split_combined_dir = os.path.join(combined_dir, split)
        
        # Brier 1x2
        grid_b_1x2 = [[os.path.join(split_single_dir, f"brier_trend_group0.png"), os.path.join(split_single_dir, f"brier_trend_group1.png")]]
        out_brier_1x2 = os.path.join(split_combined_dir, "combined_brier_1x2.png")
        combine_grid(
            grid_b_1x2, out_brier_1x2,
            title=f"Brier Score Trends — Split: {split.upper()} ({pca_status})",
            tile_w=650, tile_h=480,
            col_labels=col_labels_1x2
        )
        
        # Logloss 1x2
        grid_l_1x2 = [[os.path.join(split_single_dir, f"logloss_trend_group0.png"), os.path.join(split_single_dir, f"logloss_trend_group1.png")]]
        out_logloss_1x2 = os.path.join(split_combined_dir, "combined_logloss_1x2.png")
        combine_grid(
            grid_l_1x2, out_logloss_1x2,
            title=f"Log Loss Trends — Split: {split.upper()} ({pca_status})",
            tile_w=650, tile_h=480,
            col_labels=col_labels_1x2
        )

    # 2. 全域 3x2 大圖 (彙整 test1, test2, eval 比較)
    col_labels = ["Group: y1 == 0", "Group: y1 == 1"]
    row_labels = [s.upper() for s in ALL_SPLITS]

    grid_brier = []
    grid_logloss = []
    for split in ALL_SPLITS:
        split_single_dir = os.path.join(single_dir, split)
        row_b = []
        row_l = []
        for g in [0, 1]:
            row_b.append(os.path.join(split_single_dir, f"brier_trend_group{g}.png"))
            row_l.append(os.path.join(split_single_dir, f"logloss_trend_group{g}.png"))
        grid_brier.append(row_b)
        grid_logloss.append(row_l)

    out_brier = os.path.join(combined_dir, "combined_brier_trends_split_y_3x2.png")
    combine_grid(
        grid_brier, out_brier,
        title=f"Brier Score Trends Across Hidden Layers (Unified Y-Axis Scale) | ({pca_status})",
        tile_w=650, tile_h=480,
        row_labels=row_labels, col_labels=col_labels
    )

    out_logloss = os.path.join(combined_dir, "combined_logloss_trends_split_y_3x2.png")
    combine_grid(
        grid_logloss, out_logloss,
        title=f"Log Loss Trends Across Hidden Layers (Unified Y-Axis Scale) | ({pca_status})",
        tile_w=650, tile_h=480,
        row_labels=row_labels, col_labels=col_labels
    )


# ─── 02_Reliability_Curves_split_y (6 個模型獨自畫前後) ─────────────

def plot_single_combined_reliability(y_true, score_pre, p_cal, y1, title, save_path):
    """
    繪製單子圖: 包含 y1=0 校正前後, y1=1 校正前後之 Reliability Curves
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.5)

    def get_10bin_curve(y_true_sub, y_prob_sub):
        edges = np.linspace(0.0, 1.0, 11)
        bin_ids = np.digitize(y_prob_sub, edges)
        frac_pos, mean_pred = [], []
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
    mask_0 = (y1 == 0)
    if np.sum(mask_0) > 0:
        y_true_0, score_pre_0, p_cal_0 = y_true[mask_0], score_pre[mask_0], p_cal[mask_0]
        m_raw0 = calculate_all_metrics(y_true_0, score_pre_0)
        m_cal0 = calculate_all_metrics(y_true_0, p_cal_0)

        frac_r0, mean_r0 = get_10bin_curve(y_true_0, score_pre_0)
        v_r0 = ~np.isnan(frac_r0)
        ax.plot(mean_r0[v_r0], frac_r0[v_r0], "o--", color="red", alpha=0.35,
                label=f"y1=0 Raw (Brier: {m_raw0['brier']:.4f})")

        frac_c0, mean_c0 = get_10bin_curve(y_true_0, p_cal_0)
        v_c0 = ~np.isnan(frac_c0)
        ax.plot(mean_c0[v_c0], frac_c0[v_c0], "s-", color="red", linewidth=2.0, alpha=0.9,
                label=f"y1=0 PAVA (Brier: {m_cal0['brier']:.4f})")

    # Group y1 == 1
    mask_1 = (y1 == 1)
    if np.sum(mask_1) > 0:
        y_true_1, score_pre_1, p_cal_1 = y_true[mask_1], score_pre[mask_1], p_cal[mask_1]
        m_raw1 = calculate_all_metrics(y_true_1, score_pre_1)
        m_cal1 = calculate_all_metrics(y_true_1, p_cal_1)

        frac_r1, mean_r1 = get_10bin_curve(y_true_1, score_pre_1)
        v_r1 = ~np.isnan(frac_r1)
        ax.plot(mean_r1[v_r1], frac_r1[v_r1], "o--", color="blue", alpha=0.35,
                label=f"y1=1 Raw (Brier: {m_raw1['brier']:.4f})")

        frac_c1, mean_c1 = get_10bin_curve(y_true_1, p_cal_1)
        v_c1 = ~np.isnan(frac_c1)
        ax.plot(mean_c1[v_c1], frac_c1[v_c1], "s-", color="blue", linewidth=2.0, alpha=0.9,
                label=f"y1=1 PAVA (Brier: {m_cal1['brier']:.4f})")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("Mean Predicted Probability", fontsize=10, fontweight='bold')
    ax.set_ylabel("Fraction of Positives", fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc="upper left", fontsize=8.5)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)

def generate_02_reliability_curves_combined(calib_data, base_output_dir, pca_status):
    print("\n" + "="*80)
    print(f"【圖表 3/3】生成 02_Reliability_Curves_combined (6個模型獨立大圖) | PCA: {pca_status}")
    print("="*80)

    single_dir = os.path.join(base_output_dir, "02_Reliability_Curves_combined", "single", pca_status)
    combined_dir = os.path.join(base_output_dir, "02_Reliability_Curves_combined", "combined", pca_status)

    for model_name in ALL_MODELS:
        print(f"  ├─ 處理模型: {model_name:20s}")
        for split in ALL_SPLITS:
            split_single_dir = os.path.join(single_dir, split)
            for layer in ALL_LAYERS:
                try:
                    info = calib_data['layers'][layer][model_name][split]
                except KeyError:
                    continue

                score_pre = info['score_pre']
                p_cal = info['prob_cal_split']
                y1 = info['y1']
                y3 = info['y3']

                title = f"Reliability Curves (Split y1) | Layer {layer} | {split.upper()}"
                save_p = os.path.join(split_single_dir, model_name, f"{model_name}_layer{layer}_reliability.png")
                plot_single_combined_reliability(y3, score_pre, p_cal, y1, title, save_p)

    print("  └─ 開始生成 3×6 組合大圖...")
    col_labels = [MODEL_DISPLAY_NAMES[m] for m in ALL_MODELS]
    row_labels = [s.upper() for s in ALL_SPLITS]

    layer = ALL_LAYERS[0]
    
    grid = []
    for split in ALL_SPLITS:
        split_single_dir = os.path.join(single_dir, split)
        row = []
        for model_name in ALL_MODELS:
            p = os.path.join(split_single_dir, model_name, f"{model_name}_layer{layer}_reliability.png")
            row.append(p)
        grid.append(row)

    out_comb = os.path.join(combined_dir, f"reliability_combined_3x6_all_models.png")
    combine_grid(
        grid, out_comb,
        title=f"Reliability Curves 3x6 Overview (y3 Target) | ({pca_status})",
        tile_w=580, tile_h=520,
        row_labels=row_labels, col_labels=col_labels
    )


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    input_dir = "outputs/v2_framework/framework_calibration"
    joblib_file = os.path.join(input_dir, "calibration_data.joblib")
    base_output_dir = "results/v2_framework/plots_framework_stage2"
    pca_status = "full_1024d"

    if not os.path.exists(joblib_file):
        print(f"錯誤: 找不到校準數據 {joblib_file}。請先執行 calibrate_framework_models.py")
        sys.exit(1)

    print("=" * 80)
    print(f"階段二核心可視化圖表 (V2) — 開始繪製")
    print("=" * 80)

    calib_data = joblib.load(joblib_file)

    generate_07_joint_calibration(calib_data, base_output_dir, pca_status)
    generate_01_metrics_trends_split_y(calib_data, base_output_dir, pca_status)
    generate_02_reliability_curves_combined(calib_data, base_output_dir, pca_status)

    print("\n" + "=" * 80)
    print(f"所有指定的階段二核心圖表已繪製完成！({pca_status}) 圖片儲存於: {base_output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()
