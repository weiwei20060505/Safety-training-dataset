"""
8月6日 LLM 隱藏狀態機率校正與元評估框架 — 階段二：獨立診斷繪圖腳本
======================================================================
從 results/framework_calibration/ 讀取數據，繪製階段二可視化診斷圖表：
1. joint_calibration (07_Joint_Calibration: Histogram + Scatter 聯合校正圖) — 強制必選
2. reliability       (02_Reliability_Curves: 可靠度對比圖)
3. brier_components  (05_Brier_Components: Brier Reliability & Resolution 條形圖)
4. step_mappings     (06_Step_Mappings: PAVA 階梯映射條形圖)

支援 CLI 參數獨立單單繪製某一類圖表 (Avoid full re-running):
  python plot_framework_stage2.py --chart joint_calibration --split test2 --layer 4
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# 設定字體與風格 (英文無 missing glyph 警告)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

ALL_LAYERS = [3, 4, 5, 6]
ALL_SPLITS = ['test1', 'test2', 'eval']
ALL_MODELS = ['RootSplit_LGBM', 'Feature129_LGBM', 'YHead_MLP', 'SingleHead129_MLP']

MODEL_COLORS = {
    'RootSplit_LGBM': '#D9381E',    # Crimson Red
    'Feature129_LGBM': '#E67E22',   # Orange
    'YHead_MLP': '#2B547E',         # Deep Blue
    'SingleHead129_MLP': '#8E44AD'  # Purple
}

def plot_joint_calibration(calib_data, output_dir, layers, splits, models):
    """繪製 07_Joint_Calibration (Histogram + Scatter 聯合校正圖)"""
    print("\n[繪圖] 07_Joint_Calibration (Histogram + Scatter 聯合校正圖)...")
    dst_dir = os.path.join(output_dir, "07_Joint_Calibration")
    os.makedirs(dst_dir, exist_ok=True)

    for split in splits:
        for layer in layers:
            for model_name in models:
                try:
                    info = calib_data['layers'][layer][model_name][split]
                except KeyError:
                    continue

                s_raw = info['s_raw']
                p_cal = info['prob_cal_split']
                y2 = info['y2']

                fig, (ax_hist, ax_scatter) = plt.subplots(
                    2, 1, figsize=(7, 7.5), gridspec_kw={'height_ratios': [1, 2.2]}
                )
                fig.suptitle(f"Layer {layer} - {model_name} Joint Calibration ({split.upper()})", fontsize=13, fontweight='bold')

                # 上半部: 分數分布直方圖
                ax_hist.hist(s_raw, bins=25, alpha=0.5, color='#D9381E', label='Raw Score S', density=True)
                ax_hist.hist(p_cal, bins=25, alpha=0.5, color='#2B547E', label='PAVA Calibrated P', density=True)
                ax_hist.set_ylabel("Density", fontsize=10)
                ax_hist.set_xlim([-0.02, 1.02])
                ax_hist.legend(loc='upper right', fontsize=8.5)
                ax_hist.grid(True, alpha=0.3)

                # 下半部: 校正散點與 45 度對角線
                ax_scatter.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6, label='Perfect Calibration')
                ax_scatter.scatter(s_raw, p_cal, alpha=0.3, color='#2B547E', s=12, label='Sample Mapping')

                # 繪製 Isotonic 階梯映射
                sorted_idx = np.argsort(s_raw)
                ax_scatter.plot(s_raw[sorted_idx], p_cal[sorted_idx], color='#D9381E', lw=2.2, label='PAVA Monotone Curve')

                brier_r = info['metrics_raw']['brier']
                brier_c = info['metrics_cal_split']['brier']
                ax_scatter.set_title(f"Brier Score: {brier_r:.4f} -> {brier_c:.4f}", fontsize=11)
                ax_scatter.set_xlabel("Raw Score S", fontsize=11)
                ax_scatter.set_ylabel("Calibrated Probability P", fontsize=11)
                ax_scatter.set_xlim([-0.02, 1.02])
                ax_scatter.set_ylim([-0.02, 1.02])
                ax_scatter.legend(loc='upper left', fontsize=8.5)
                ax_scatter.grid(True, alpha=0.3)

                plt.tight_layout()
                save_name = f"joint_calibration_layer_{layer}_{model_name}_{split}.png"
                fig.savefig(os.path.join(dst_dir, save_name), dpi=200, bbox_inches='tight')
                plt.close(fig)

    print(f"  └─ 已完成 Joint Calibration 圖片輸出: {dst_dir}")

def plot_reliability_curves(calib_data, output_dir, layers, splits, models):
    """繪製 02_Reliability_Curves (1x4 橫向與單圖對比)"""
    print("\n[繪圖] 02_Reliability_Curves (可靠度對比圖)...")
    dst_dir = os.path.join(output_dir, "02_Reliability_Curves")
    os.makedirs(dst_dir, exist_ok=True)

    for split in splits:
        fig, axes = plt.subplots(1, 4, figsize=(22, 5), sharey=True)
        fig.suptitle(f"Reliability Curves Across Layers 3~6 (PAVA Calibrated vs Raw - {split.upper()})", fontsize=15, fontweight='bold', y=1.02)

        for idx, layer in enumerate(layers):
            ax = axes[idx]
            ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6, label='Perfect')

            for model_name in models:
                try:
                    info = calib_data['layers'][layer][model_name][split]
                except KeyError:
                    continue

                s_raw = info['s_raw']
                p_cal = info['prob_cal_split']
                y2 = info['y2']
                color = MODEL_COLORS.get(model_name, '#555555')

                # 計算 10 Bins
                edges = np.linspace(0, 1, 11)
                bin_ids = np.digitize(p_cal, edges)
                mean_p, frac_y = [], []
                for b in range(1, 11):
                    mask = (bin_ids == b)
                    if np.sum(mask) > 0:
                        mean_p.append(np.mean(p_cal[mask]))
                        frac_y.append(np.mean(y2[mask]))

                brier_c = info['metrics_cal_split']['brier']
                ax.plot(mean_p, frac_y, 'o-', color=color, lw=2, label=f"{model_name} (Brier={brier_c:.3f})")

            ax.set_title(f"Layer {layer}", fontsize=12, fontweight='bold')
            ax.set_xlabel("Mean Predicted Probability", fontsize=10)
            if idx == 0:
                ax.set_ylabel("Fraction of Positives", fontsize=10)
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(os.path.join(dst_dir, f"reliability_1x4_{split}.png"), dpi=200, bbox_inches='tight')
        plt.close(fig)

    print(f"  └─ 已完成 Reliability Curves 圖片輸出: {dst_dir}")

def plot_brier_components(df_summary, output_dir, layers, splits, models):
    """繪製 05_Brier_Components (Reliability & Resolution 分解條形圖)"""
    print("\n[繪圖] 05_Brier_Components (Brier Reliability & Resolution 分解)...")
    dst_dir = os.path.join(output_dir, "05_Brier_Components")
    os.makedirs(dst_dir, exist_ok=True)

    for split in splits:
        df_sp = df_summary[df_summary['split'] == split]

        fig, (ax_rel, ax_res) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Brier Components Decomposition ({split.upper()})", fontsize=14, fontweight='bold')

        # Reliability (Lower is better)
        x = np.arange(len(layers))
        width = 0.18
        for m_idx, model_name in enumerate(models):
            df_m = df_sp[df_sp['model'] == model_name]
            rel_vals = [df_m[df_m['layer'] == l]['reliability_cal'].values[0] if len(df_m[df_m['layer'] == l]) > 0 else 0 for l in layers]
            ax_rel.bar(x + m_idx * width, rel_vals, width, label=model_name, color=MODEL_COLORS.get(model_name))

        ax_rel.set_title("Reliability (Calibration Error - Lower is Better)", fontsize=11)
        ax_rel.set_xticks(x + width * 1.5)
        ax_rel.set_xticklabels([f"Layer {l}" for l in layers])
        ax_rel.legend(fontsize=8.5)
        ax_rel.grid(axis='y', alpha=0.4)

        # Resolution (Higher is better)
        for m_idx, model_name in enumerate(models):
            df_m = df_sp[df_sp['model'] == model_name]
            res_vals = [df_m[df_m['layer'] == l]['resolution_cal'].values[0] if len(df_m[df_m['layer'] == l]) > 0 else 0 for l in layers]
            ax_res.bar(x + m_idx * width, res_vals, width, label=model_name, color=MODEL_COLORS.get(model_name))

        ax_res.set_title("Resolution (Discrimination Power - Higher is Better)", fontsize=11)
        ax_res.set_xticks(x + width * 1.5)
        ax_res.set_xticklabels([f"Layer {l}" for l in layers])
        ax_res.legend(fontsize=8.5)
        ax_res.grid(axis='y', alpha=0.4)

        plt.tight_layout()
        fig.savefig(os.path.join(dst_dir, f"brier_components_{split}.png"), dpi=200, bbox_inches='tight')
        plt.close(fig)

    print(f"  └─ 已完成 Brier Components 圖片輸出: {dst_dir}")

def plot_step_mappings(calib_data, output_dir, layers, splits, models):
    """繪製 06_Step_Mappings (PAVA 階梯映射條形圖)"""
    print("\n[繪圖] 06_Step_Mappings (PAVA 階梯映射圖)...")
    dst_dir = os.path.join(output_dir, "06_Step_Mappings")
    os.makedirs(dst_dir, exist_ok=True)

    for split in splits:
        for layer in layers:
            for model_name in models:
                try:
                    info = calib_data['layers'][layer][model_name][split]
                except KeyError:
                    continue

                s_raw = info['s_raw']
                p_cal = info['prob_cal_split']

                edges = np.linspace(0.0, 1.0, 11)
                bin_labels = [f'{i/10:.1f}-{(i+1)/10:.1f}' for i in range(10)]
                bin_ids = np.digitize(s_raw, edges)

                mapped_means = []
                for b in range(1, 11):
                    mask = (bin_ids == b)
                    if np.sum(mask) > 0:
                        mapped_means.append(np.mean(p_cal[mask]))
                    else:
                        mapped_means.append(0.0)

                fig, ax = plt.subplots(figsize=(7, 4.5))
                bars = ax.bar(np.arange(10), mapped_means, color='#55A868', edgecolor='black', alpha=0.85)

                for bar in bars:
                    h = bar.get_height()
                    if h > 0:
                        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

                ax.set_xticks(np.arange(10))
                ax.set_xticklabels(bin_labels, rotation=30, fontsize=8.5)
                ax.set_ylim([0, 1.1])
                ax.set_title(f"Layer {layer} - {model_name} PAVA Step Mapping ({split.upper()})", fontsize=12, fontweight='bold')
                ax.set_xlabel("Raw Score Bins", fontsize=10)
                ax.set_ylabel("Calibrated Probability", fontsize=10)
                ax.grid(axis='y', alpha=0.3)

                plt.tight_layout()
                fig.savefig(os.path.join(dst_dir, f"step_mapping_layer_{layer}_{model_name}_{split}.png"), dpi=180, bbox_inches='tight')
                plt.close(fig)

    print(f"  └─ 已完成 Step Mappings 圖片輸出: {dst_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="LLM Safety Probe Pipeline - Stage 2: Calibration Diagnostics Plotting",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--chart',
        choices=['all', 'joint_calibration', 'reliability', 'brier_components', 'step_mappings'],
        default='all',
        help="選擇要繪製的圖表類型 (預設: all；包含 mandatory 之 joint_calibration)"
    )
    parser.add_argument(
        '--layer',
        choices=['3', '4', '5', '6', 'all'],
        default='all',
        help="選擇隱藏特徵層 (預設: all)"
    )
    parser.add_argument(
        '--split',
        choices=['test1', 'test2', 'eval', 'all'],
        default='all',
        help="選擇測試集 (預設: all)"
    )
    parser.add_argument(
        '--model',
        choices=['RootSplit_LGBM', 'Feature129_LGBM', 'YHead_MLP', 'SingleHead129_MLP', 'all'],
        default='all',
        help="選擇模型 (預設: all)"
    )
    args = parser.parse_args()

    input_dir = "results/framework_calibration"
    output_dir = "results/plots_framework_stage2"
    os.makedirs(output_dir, exist_ok=True)

    summary_csv = os.path.join(input_dir, "framework_calibration_summary.csv")
    joblib_file = os.path.join(input_dir, "calibration_data.joblib")

    if not os.path.exists(summary_csv) or not os.path.exists(joblib_file):
        print(f"錯誤: 找不到校準數據。請先執行 python calibrate_framework_models.py。")
        sys.exit(1)

    df_summary = pd.read_csv(summary_csv)
    calib_data = joblib.load(joblib_file)

    layers = ALL_LAYERS if args.layer == 'all' else [int(args.layer)]
    splits = ALL_SPLITS if args.split == 'all' else [args.split]
    models = ALL_MODELS if args.model == 'all' else [args.model]

    print("=" * 80)
    print("階段二獨立可視化診斷繪圖 — 開始執行")
    print(f"  ├─ 圖表類型: {args.chart}")
    print(f"  ├─ 隱藏層數: {layers}")
    print(f"  ├─ 數據集: {splits}")
    print(f"  └─ 模型: {models}")
    print("=" * 80)

    if args.chart in ['all', 'joint_calibration']:
        plot_joint_calibration(calib_data, output_dir, layers, splits, models)

    if args.chart in ['all', 'reliability']:
        plot_reliability_curves(calib_data, output_dir, layers, splits, models)

    if args.chart in ['all', 'brier_components']:
        plot_brier_components(df_summary, output_dir, layers, splits, models)

    if args.chart in ['all', 'step_mappings']:
        plot_step_mappings(calib_data, output_dir, layers, splits, models)

    print("\n" + "=" * 80)
    print(f"階段二繪圖完成！所有圖表儲存於: {output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()
