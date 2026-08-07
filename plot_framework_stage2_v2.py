"""
8月6日 LLM 隱藏狀態機率校正與元評估框架 — 階段二核心可視化圖表 (V2)
======================================================================
針對使用者指定的 3 大圖表類型繪製高清圖片，完全還原 results/plots/combined 樣式：
1. 07_Joint_Calibration：Histogram and Scatter Plot of Confidence 雙 Y 軸聯合校正圖 (與附圖完全一致)
2. 01_Metrics_Trends_split_y：Brier Score 與 Log Loss 隨層數變化趨勢圖 (大圖 Y 軸強制統一刻度範圍)
3. 02_Reliability_Curves_combined (02_Reliability_Curves_split_y)：4 個模型獨自繪製校正前後 y1=0 / y1=1 組合大圖 (共 4 張大圖)

支援獨立選用 `--chart [all|joint_calibration|trends_split_y|reliability_combined]`
"""

import os
import sys
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

ALL_LAYERS = [3, 4, 5, 6]
ALL_SPLITS = ['test1', 'test2', 'eval']
ALL_MODELS = ['RootSplit_LGBM', 'Feature129_LGBM', 'YHead_MLP', 'SingleHead129_MLP']
MODEL_DISPLAY_NAMES = {
    'RootSplit_LGBM': 'RootSplit-LGBM',
    'Feature129_LGBM': 'Feature129-LGBM',
    'YHead_MLP': 'YHead-MLP',
    'SingleHead129_MLP': 'SingleHead129-MLP'
}
MODEL_COLORS = {
    'RootSplit_LGBM': '#4C72B0',    # Blue
    'Feature129_LGBM': '#55A868',   # Green
    'YHead_MLP': '#C44E52',         # Red
    'SingleHead129_MLP': '#8172B3'  # Purple
}
MODEL_MARKERS = {
    'RootSplit_LGBM': 'o',
    'Feature129_LGBM': 's',
    'YHead_MLP': '^',
    'SingleHead129_MLP': 'D'
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
    完美還原用戶附圖:
    - 雙 Y 軸 (左: Frequency, 右: Accuracy)
    - 正樣本 (y2==1) 綠色長條, 負樣本 (y2==0) 紅色長條
    - 黑色對角線 Perfect calibration
    - 灰色階梯線 Isotonic regression
    - 藍色圓圈點 Bin accuracy
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

def generate_07_joint_calibration(calib_data, base_output_dir):
    print("\n" + "="*80)
    print("【圖表 1/3】生成 07_Joint_Calibration (Histogram and Scatter Plot of Confidence)")
    print("="*80)

    single_dir = os.path.join(base_output_dir, "07_Joint_Calibration", "single")
    combined_dir = os.path.join(base_output_dir, "07_Joint_Calibration", "combined")

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

    print("  └─ 單張圖繪製完成，開始拼接 2×4 組合大圖...")
    # 拼接大圖 (2 列: group0, group1 × 4 行: Layer 3, 4, 5, 6)
    col_labels = [f"Layer {l}" for l in ALL_LAYERS]
    row_labels = ["y1 == 0", "y1 == 1"]

    for split in ALL_SPLITS:
        for model_name in ALL_MODELS:
            grid = []
            for g in [0, 1]:
                row = []
                for layer in ALL_LAYERS:
                    p = os.path.join(single_dir, split, f"joint_cal_layer{layer}_{model_name}_group{g}.png")
                    row.append(p)
                grid.append(row)

            out_comb = os.path.join(combined_dir, f"joint_cal_combined_{model_name}_{split}_2x4.png")
            combine_grid(
                grid, out_comb,
                title=f"Joint Calibration Overview — Model: {MODEL_DISPLAY_NAMES[model_name]} | Split: {split.upper()}",
                tile_w=650, tile_h=650,
                row_labels=row_labels,
                col_labels=col_labels
            )


# ─── 01_Metrics_Trends_split_y (Y 軸全域強制統一) ────────────────────────────

def generate_01_metrics_trends_split_y(calib_data, base_output_dir):
    print("\n" + "="*80)
    print("【圖表 2/3】生成 01_Metrics_Trends_split_y (Brier & LogLoss 趨勢圖 - 統一 Y 軸)")
    print("="*80)

    single_dir = os.path.join(base_output_dir, "01_Metrics_Trends_split_y", "single")
    combined_dir = os.path.join(base_output_dir, "01_Metrics_Trends_split_y", "combined")

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
        for g in [0, 1]:
            df_sg = df[(df['split'] == split) & (df['group'] == g)]

            # 1. Brier Trend
            fig, ax = plt.subplots(figsize=(7.5, 5.5))
            for model_name in ALL_MODELS:
                df_m = df_sg[df_sg['model'] == model_name].sort_values('layer')
                if df_m.empty:
                    continue
                ax.plot(
                    df_m['layer'], df_m['brier'],
                    label=MODEL_DISPLAY_NAMES[model_name],
                    color=MODEL_COLORS[model_name],
                    marker=MODEL_MARKERS[model_name],
                    markersize=7, linewidth=2.0, alpha=0.9
                )
            ax.set_xticks(ALL_LAYERS)
            ax.set_ylim(brier_ylim)
            ax.set_xlabel('Hidden Layer (3~6)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Brier Score (Lower is Better)', fontsize=11, fontweight='bold')
            ax.set_title(f'Brier Score Trend (Group: y1 == {g}) | Split: {split.upper()}', fontsize=12, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.legend(loc='upper right', fontsize=9)
            plt.tight_layout()

            save_brier = os.path.join(single_dir, f"brier_trend_{split}_group{g}.png")
            os.makedirs(os.path.dirname(save_brier), exist_ok=True)
            fig.savefig(save_brier, dpi=180, bbox_inches='tight')
            plt.close(fig)

            # 2. Log Loss Trend
            fig, ax = plt.subplots(figsize=(7.5, 5.5))
            for model_name in ALL_MODELS:
                df_m = df_sg[df_sg['model'] == model_name].sort_values('layer')
                if df_m.empty:
                    continue
                ax.plot(
                    df_m['layer'], df_m['logloss'],
                    label=MODEL_DISPLAY_NAMES[model_name],
                    color=MODEL_COLORS[model_name],
                    marker=MODEL_MARKERS[model_name],
                    markersize=7, linewidth=2.0, alpha=0.9
                )
            ax.set_xticks(ALL_LAYERS)
            ax.set_ylim(logloss_ylim)
            ax.set_xlabel('Hidden Layer (3~6)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Log Loss (Lower is Better)', fontsize=11, fontweight='bold')
            ax.set_title(f'Log Loss Trend (Group: y1 == {g}) | Split: {split.upper()}', fontsize=12, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.legend(loc='upper right', fontsize=9)
            plt.tight_layout()

            save_logloss = os.path.join(single_dir, f"logloss_trend_{split}_group{g}.png")
            os.makedirs(os.path.dirname(save_logloss), exist_ok=True)
            fig.savefig(save_logloss, dpi=180, bbox_inches='tight')
            plt.close(fig)

    # 拼接大圖 (3 列: test1, test2, eval × 2 行: group 0, group 1)
    col_labels = ["Group: y1 == 0", "Group: y1 == 1"]
    row_labels = [s.upper() for s in ALL_SPLITS]

    grid_brier = []
    grid_logloss = []
    for split in ALL_SPLITS:
        row_b = []
        row_l = []
        for g in [0, 1]:
            row_b.append(os.path.join(single_dir, f"brier_trend_{split}_group{g}.png"))
            row_l.append(os.path.join(single_dir, f"logloss_trend_{split}_group{g}.png"))
        grid_brier.append(row_b)
        grid_logloss.append(row_l)

    out_brier = os.path.join(combined_dir, "combined_brier_trends_split_y_3x2.png")
    combine_grid(
        grid_brier, out_brier,
        title="Brier Score Trends Across Hidden Layers (Unified Y-Axis Scale)",
        tile_w=650, tile_h=480,
        row_labels=row_labels, col_labels=col_labels
    )

    out_logloss = os.path.join(combined_dir, "combined_logloss_trends_split_y_3x2.png")
    combine_grid(
        grid_logloss, out_logloss,
        title="Log Loss Trends Across Hidden Layers (Unified Y-Axis Scale)",
        tile_w=650, tile_h=480,
        row_labels=row_labels, col_labels=col_labels
    )


# ─── 02_Reliability_Curves_split_y (4 個模型獨自畫前後，大圖共 4 張) ─────────────

def plot_single_combined_reliability(y_true, score_pre, p_cal, y1, title, save_path):
    """
    繪製單子圖: 包含 y1=0 校正前後, y1=1 校正前後之 Reliability Curves
    與 02_Reliability_Curves_combined 格式完全對齊
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

def generate_02_reliability_curves_combined(calib_data, base_output_dir):
    print("\n" + "="*80)
    print("【圖表 3/3】生成 02_Reliability_Curves_combined (4個模型獨立大圖: 3splits × 4layers)")
    print("="*80)

    single_dir = os.path.join(base_output_dir, "02_Reliability_Curves_combined", "single")
    combined_dir = os.path.join(base_output_dir, "02_Reliability_Curves_combined", "combined")

    for model_name in ALL_MODELS:
        print(f"  ├─ 處理模型: {model_name:20s} (生成 12 張單圖與 1 張 3×4 大圖)")
        for split in ALL_SPLITS:
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
                save_p = os.path.join(single_dir, model_name, f"{model_name}_{split}_layer{layer}_reliability.png")
                plot_single_combined_reliability(y3, score_pre, p_cal, y1, title, save_p)

        # 為該模型拼接 3 列 (test1, test2, eval) × 4 行 (Layer 3, 4, 5, 6) 的獨自大圖 (共有 4 張大圖)
        col_labels = [f"Layer {l}" for l in ALL_LAYERS]
        row_labels = [s.upper() for s in ALL_SPLITS]

        grid = []
        for split in ALL_SPLITS:
            row = []
            for layer in ALL_LAYERS:
                p = os.path.join(single_dir, model_name, f"{model_name}_{split}_layer{layer}_reliability.png")
                row.append(p)
            grid.append(row)

        out_comb = os.path.join(combined_dir, f"y3_{model_name}_combined_reliability_3x4.png")
        combine_grid(
            grid, out_comb,
            title=f"Reliability Curves Before vs After Calibration — Model: {MODEL_DISPLAY_NAMES[model_name]} (y3 Target)",
            tile_w=580, tile_h=520,
            row_labels=row_labels, col_labels=col_labels
        )


def main():
    parser = argparse.ArgumentParser(
        description="LLM Safety Probe Pipeline - Stage 2 Core Plotting Suite (V2)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--chart',
        choices=['all', 'joint_calibration', 'trends_split_y', 'reliability_combined'],
        default='all',
        help="選擇要繪製的圖表類型 (預設: all)"
    )
    args = parser.parse_args()

    input_dir = "results/framework_calibration"
    joblib_file = os.path.join(input_dir, "calibration_data.joblib")
    base_output_dir = "results/plots_framework_stage2"

    if not os.path.exists(joblib_file):
        print(f"錯誤: 找不到校準數據 {joblib_file}。請先執行 calibrate_framework_models.py")
        sys.exit(1)

    print("=" * 80)
    print(f"階段二核心可視化圖表 (V2) — 開始繪製圖表類型: {args.chart}")
    print("=" * 80)

    calib_data = joblib.load(joblib_file)

    if args.chart in ['all', 'joint_calibration']:
        generate_07_joint_calibration(calib_data, base_output_dir)

    if args.chart in ['all', 'trends_split_y']:
        generate_01_metrics_trends_split_y(calib_data, base_output_dir)

    if args.chart in ['all', 'reliability_combined']:
        generate_02_reliability_curves_combined(calib_data, base_output_dir)

    print("\n" + "=" * 80)
    print(f"所有指定的階段二核心圖表已繪製完成！圖片儲存於: {base_output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()
