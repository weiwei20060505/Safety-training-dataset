import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# Set font for proper rendering
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# Path configuration
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE_PKL = os.path.join(BASE_DIR, "cache/v1_baseline/calibration/without_pca/calibrated_predictions.pkl")
OUTPUT_DIR = os.path.join(BASE_DIR, "results/v1_baseline/plots_v1_joint_calibration_split_y1")

MODEL_DISPLAY_NAMES = {
    'SGD': 'SGD',
    'MLP': 'MLP',
    'LGB': 'LightGBM',
    'LR': 'Logistic Regression',
    'RF': 'Random Forest'
}

ALL_MODELS = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
ALL_SPLITS = ['test1', 'test2', 'eval']
ALL_LAYERS = [1, 2, 3, 4, 5, 6]

def _load_font(size: int = 20):
    candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\Arial.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()

def combine_grid(image_paths_grid, output_path, title="", tile_w=650, tile_h=650, row_labels=None, col_labels=None):
    """
    Combines a grid of image files into a single composite image with headers & labels.
    """
    n_rows = len(image_paths_grid)
    n_cols = len(image_paths_grid[0]) if n_rows > 0 else 0
    if n_rows == 0 or n_cols == 0:
        return

    header_h = 70 if title else 20
    col_label_h = 35 if col_labels else 0
    row_label_w = 120 if row_labels else 0
    margin = 10

    total_w = row_label_w + n_cols * tile_w + (n_cols + 1) * margin
    total_h = header_h + col_label_h + n_rows * tile_h + (n_rows + 1) * margin

    canvas = Image.new('RGB', (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    font_title = _load_font(24)
    font_label = _load_font(18)

    if title:
        draw.text((row_label_w + margin, 15), title, fill=(0, 0, 0), font=font_title)

    if col_labels:
        for c, clab in enumerate(col_labels):
            x = row_label_w + margin + c * (tile_w + margin) + tile_w // 2
            y = header_h + 5
            draw.text((x, y), str(clab), fill=(50, 50, 50), font=font_label, anchor="mm")

    if row_labels:
        for r, rlab in enumerate(row_labels):
            x = margin + row_label_w // 2
            y = header_h + col_label_h + margin + r * (tile_h + margin) + tile_h // 2
            draw.text((x, y), str(rlab), fill=(50, 50, 50), font=font_label, anchor="mm")

    for r in range(n_rows):
        for c in range(n_cols):
            img_path = image_paths_grid[r][c]
            x_pos = row_label_w + margin + c * (tile_w + margin)
            y_pos = header_h + col_label_h + margin + r * (tile_h + margin)

            if os.path.exists(img_path):
                try:
                    tile = Image.open(img_path).convert('RGB')
                    tile = tile.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
                    canvas.paste(tile, (x_pos, y_pos))
                except Exception as e:
                    draw.rectangle([x_pos, y_pos, x_pos + tile_w, y_pos + tile_h], outline=(200, 200, 200), fill=(245, 245, 245))
                    draw.text((x_pos + 20, y_pos + tile_h // 2), f"Error: {e}", fill=(255, 0, 0), font=font_label)
            else:
                draw.rectangle([x_pos, y_pos, x_pos + tile_w, y_pos + tile_h], outline=(200, 200, 200), fill=(245, 245, 245))
                draw.text((x_pos + 20, y_pos + tile_h // 2), "N/A", fill=(150, 150, 150), font=font_label)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas.save(output_path, quality=95)
    print(f"Saved grid image: {output_path}")

def plot_single_joint_calibration(pre_scores, post_scores, y_true, title, save_path):
    """
    Plots a single Histogram and Scatter Plot of Confidence with dual Y-axes.
    """
    fig, ax1 = plt.subplots(figsize=(7.5, 7.5))

    ax1.set_xlabel('Confidence', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')

    mask_pos = (y_true == 1)
    mask_neg = (y_true == 0)

    bins = np.linspace(0.0, 1.0, 21)
    ax1.hist(pre_scores[mask_pos], bins=bins, color='#74c476', alpha=0.75, label='Correct (y3=1)', zorder=2)
    ax1.hist(pre_scores[mask_neg], bins=bins, color='#fb6a4a', alpha=0.75, label='Incorrect (y3=0)', zorder=1)
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='blue', fontsize=11, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='blue')

    # 1. Perfect calibration line
    ax2.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=1.5)

    # 2. Isotonic regression step curve
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

    # 3. Empirical Bin accuracy scatter points
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

    plt.title(title, fontsize=11, fontweight='bold', pad=12)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)

import joblib

def main():
    print(f"Loading predictions cache from: {CACHE_PKL}")
    data = joblib.load(CACHE_PKL)

    # We evaluate task y3 (consistency label)
    y2_data = data['y2']

    single_dir = os.path.join(OUTPUT_DIR, "single")
    combined_dir = os.path.join(OUTPUT_DIR, "combined")

    print("\nGenerating single Joint Calibration plots split by y1 = 0 and y1 = 1...")
    count = 0
    for layer in ALL_LAYERS:
        for split in ALL_SPLITS:
            for model_name in ALL_MODELS:
                try:
                    info = y2_data[layer]['splits'][split][model_name]
                except KeyError:
                    continue

                score_pre = info['score_pre']
                p_cal = info['y_prob']  # Calibrated probability
                y1 = info['y1']
                y3 = info['y3']

                # 1. Group y1 == 0
                mask0 = (y1 == 0)
                if np.sum(mask0) > 0:
                    title0 = f"V1 Baseline Joint Calibration (Task: Y3, Layer: {layer}, Model: {model_name}, Split: {split.upper()})\nSubgroup: y1 == 0 (Benign / Non-harmful Prompt)"
                    save0 = os.path.join(single_dir, f"layer{layer}", split, f"joint_cal_L{layer}_{model_name}_{split}_y1_0.png")
                    plot_single_joint_calibration(score_pre[mask0], p_cal[mask0], y3[mask0], title0, save0)

                # 2. Group y1 == 1
                mask1 = (y1 == 1)
                if np.sum(mask1) > 0:
                    title1 = f"V1 Baseline Joint Calibration (Task: Y3, Layer: {layer}, Model: {model_name}, Split: {split.upper()})\nSubgroup: y1 == 1 (Harmful Prompt)"
                    save1 = os.path.join(single_dir, f"layer{layer}", split, f"joint_cal_L{layer}_{model_name}_{split}_y1_1.png")
                    plot_single_joint_calibration(score_pre[mask1], p_cal[mask1], y3[mask1], title1, save1)

                count += 2

    print(f"Generated {count} single plot images.")

    # Generate combined grid images for Layer 6 (and Layer 4, Layer 5 if desired)
    for layer in [4, 5, 6]:
        print(f"\nGenerating combined grid images for Layer {layer}...")
        col_labels_5models = [MODEL_DISPLAY_NAMES[m] for m in ALL_MODELS]
        row_labels_splits = [s.upper() for s in ALL_SPLITS]

        # 1. Grid 3x5 for Group y1 == 0 & y1 == 1 (3 Splits x 5 Models)
        for g in [0, 1]:
            grid = []
            for split in ALL_SPLITS:
                row = []
                for model_name in ALL_MODELS:
                    p = os.path.join(single_dir, f"layer{layer}", split, f"joint_cal_L{layer}_{model_name}_{split}_y1_{g}.png")
                    row.append(p)
                grid.append(row)

            g_name = "y1=0 (Benign Prompt)" if g == 0 else "y1=1 (Harmful Prompt)"
            out_comb = os.path.join(combined_dir, f"layer{layer}", f"joint_cal_overview_L{layer}_y1_{g}.png")
            combine_grid(
                grid, out_comb,
                title=f"V1 Baseline Joint Calibration Overview (Layer {layer}, {g_name})",
                tile_w=650, tile_h=650,
                row_labels=row_labels_splits,
                col_labels=col_labels_5models
            )

        # 2. Grid 2x3 per Model (Row: y1=0 vs y1=1, Col: test1, test2, eval)
        col_labels_splits = [s.upper() for s in ALL_SPLITS]
        row_labels_y1 = ["y1 == 0 (Benign)", "y1 == 1 (Harmful)"]
        for model_name in ALL_MODELS:
            grid_2x3 = []
            for g in [0, 1]:
                row = []
                for split in ALL_SPLITS:
                    p = os.path.join(single_dir, f"layer{layer}", split, f"joint_cal_L{layer}_{model_name}_{split}_y1_{g}.png")
                    row.append(p)
                grid_2x3.append(row)

            out_comb_2x3 = os.path.join(combined_dir, f"layer{layer}", f"joint_cal_comparison_L{layer}_{model_name}.png")
            combine_grid(
                grid_2x3, out_comb_2x3,
                title=f"V1 Baseline Joint Calibration (y1=0 vs y1=1) — Layer {layer}, Model: {MODEL_DISPLAY_NAMES[model_name]}",
                tile_w=650, tile_h=650,
                row_labels=row_labels_y1,
                col_labels=col_labels_splits
            )

        # 3. Grid 2x5 per Split (Row: y1=0 vs y1=1, Col: 5 Models)
        for split in ALL_SPLITS:
            grid_2x5 = []
            for g in [0, 1]:
                row = []
                for model_name in ALL_MODELS:
                    p = os.path.join(single_dir, f"layer{layer}", split, f"joint_cal_L{layer}_{model_name}_{split}_y1_{g}.png")
                    row.append(p)
                grid_2x5.append(row)

            out_comb_2x5 = os.path.join(combined_dir, f"layer{layer}", f"joint_cal_comparison_L{layer}_{split}.png")
            combine_grid(
                grid_2x5, out_comb_2x5,
                title=f"V1 Baseline Joint Calibration (y1=0 vs y1=1) — Layer {layer}, Split: {split.upper()}",
                tile_w=650, tile_h=650,
                row_labels=row_labels_y1,
                col_labels=col_labels_5models
            )

    print("\nAll V1 Joint Calibration plots successfully generated!")

if __name__ == '__main__':
    main()
