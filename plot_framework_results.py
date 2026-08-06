"""
8月6日 LLM 隱藏狀態機率校正與元評估框架 — 獨立繪圖腳本
======================================================================
從 results/framework_training/ 讀取數據與評估結果，繪製：
1. 1×4 ROC 曲線組合圖 (Layer 3, 4, 5, 6 每張圖有 4 種模型在 Val_Set 的 ROC 曲線)
2. 6 指標 Model Comparison 條形圖 (比照使用者照片格式：Accuracy, Bal Acc, Precision, Recall, F1, AUC)

所有圖表儲存於獨立資料夾: results/plots_framework/
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# 設定中文字體與美化風格
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'PMingLiU', 'DFKai-SB', 'DejaVu Sans', 'sans-serif']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def main():
    print("=" * 80)
    print("開始執行獨立繪圖腳本 — 繪製 1×4 ROC 組合圖與 6 指標 Model Comparison 條形圖")
    print("=" * 80)

    input_dir = "results/framework_training"
    output_dir = "results/plots_framework"
    os.makedirs(output_dir, exist_ok=True)

    summary_csv = os.path.join(input_dir, "framework_evaluation_summary.csv")
    preds_file = os.path.join(input_dir, "val_predictions.joblib")

    if not os.path.exists(summary_csv) or not os.path.exists(preds_file):
        print(f"錯誤: 找不到訓練結果檔案。請先執行 python train_framework_models.py。")
        sys.exit(1)

    print(f"[1] 讀取訓練結果: {summary_csv} 與 {preds_file} ...")
    df_summary = pd.read_csv(summary_csv)
    val_preds = joblib.load(preds_file)

    layers = [3, 4, 5, 6]
    models = ["RootSplit_LGBM", "Feature129_LGBM", "YHead_MLP", "SingleHead129_MLP"]

    # 4 種模型的顏色配置
    colors_dict = {
        "RootSplit_LGBM": "#D9381E",    # Crimson Red
        "Feature129_LGBM": "#E67E22",   # Orange / Amber
        "YHead_MLP": "#2B547E",         # Deep Blue
        "SingleHead129_MLP": "#8E44AD"  # Purple
    }

    model_display_names = {
        "RootSplit_LGBM": "RootSplit-LGBM",
        "Feature129_LGBM": "Feature129-LGBM",
        "YHead_MLP": "YHead-MLP",
        "SingleHead129_MLP": "SingleHead129-MLP"
    }

    # ================= 任務 1: 畫 1x4 橫向 ROC 大圖 (val_set) =================
    print("\n[2] 繪製 1×4 ROC 組合圖 (Layer 3 ~ 6) ...")
    fig, axes = plt.subplots(1, 4, figsize=(22, 5), sharey=True)
    fig.suptitle("Validation Set ROC Curves Across Layers (4 Model Strategies)", fontsize=16, fontweight='bold', y=1.02)

    for idx, layer in enumerate(layers):
        ax = axes[idx]
        layer_data = val_preds['layers'].get(layer, {})

        for model_name in models:
            if model_name in layer_data:
                fpr = layer_data[model_name]['fpr']
                tpr = layer_data[model_name]['tpr']
                auc_val = layer_data[model_name]['auc']
                color = colors_dict.get(model_name, '#555555')
                disp_name = model_display_names.get(model_name, model_name)

                ax.plot(fpr, tpr, color=color, lw=2.2, label=f"{disp_name} (AUC={auc_val:.3f})")

        ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.2, label='Chance')
        ax.set_title(f"Layer {layer}", fontsize=13, fontweight='bold')
        ax.set_xlabel("False Positive Rate", fontsize=11)
        if idx == 0:
            ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.legend(loc='lower right', fontsize=8.5, frameon=True)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    roc_1x4_path = os.path.join(output_dir, "roc_curves_1x4_val.png")
    plt.savefig(roc_1x4_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  └─ 成功儲存 1×4 ROC 組合圖: {roc_1x4_path}")

    # ================= 任務 2: 畫 Model Comparison 條形圖 (比照照片格式: 6 個指標) =================
    print("\n[3] 繪製 6 指標 Model Comparison 條形圖...")

    metrics_keys = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'auc']
    metrics_titles = {
        'accuracy': 'Accuracy',
        'balanced_accuracy': 'Bal Acc',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1': 'F1 Score',
        'auc': 'ROC AUC'
    }

    # 2.1 逐層繪製獨立的 1x6 條形對比圖
    for layer in layers:
        df_layer = df_summary[(df_summary['layer'] == layer) & (df_summary['dataset'] == 'Val_Set')]

        fig, axes = plt.subplots(1, 6, figsize=(24, 4.2))
        fig.suptitle(f"Layer {layer} - Y2 Model Performance Comparison (Val_Set)", fontsize=15, fontweight='bold', y=1.03)

        for m_idx, m_key in enumerate(metrics_keys):
            ax = axes[m_idx]
            scores = []
            bar_colors = []
            bar_labels = []

            for model_name in models:
                row = df_layer[df_layer['model'] == model_name]
                if len(row) > 0:
                    val = row[m_key].values[0]
                    scores.append(val)
                else:
                    scores.append(0.0)
                bar_colors.append(colors_dict.get(model_name, '#555555'))
                # 簡短 X 軸標籤: LGB(Root), LGB(129), MLP(YHead), MLP(129)
                short_label = model_name.replace("RootSplit_LGBM", "LGB(Root)").replace("Feature129_LGBM", "LGB(129)").replace("YHead_MLP", "MLP(YHead)").replace("SingleHead129_MLP", "MLP(129)")
                bar_labels.append(short_label)

            x_pos = np.arange(len(models))
            bars = ax.bar(x_pos, scores, color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.7)

            # 標註具體數值
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.annotate(f"{h:.3f}",
                                xy=(bar.get_x() + bar.get_width() / 2, h),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=8.5, fontweight='bold')

            ax.set_xticks(x_pos)
            ax.set_xticklabels(bar_labels, rotation=25, fontsize=8)
            ax.set_ylim([0, 1.1])
            ax.set_title(metrics_titles[m_key], fontsize=11, fontweight='bold')
            ax.grid(axis='y', linestyle='--', alpha=0.5)

        plt.tight_layout()
        layer_comp_path = os.path.join(output_dir, f"model_comparison_layer_{layer}.png")
        plt.savefig(layer_comp_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ├─ 儲存 Layer {layer} Model Comparison: {layer_comp_path}")

    # 2.2 畫 4x6 跨層數全景條形對比總圖
    fig, axes = plt.subplots(4, 6, figsize=(24, 15), sharex=True, sharey=True)
    fig.suptitle("Model Performance Comparison Across Layers 3~6 (Val_Set)", fontsize=18, fontweight='bold', y=0.99)

    for row_idx, layer in enumerate(layers):
        df_layer = df_summary[(df_summary['layer'] == layer) & (df_summary['dataset'] == 'Val_Set')]

        for col_idx, m_key in enumerate(metrics_keys):
            ax = axes[row_idx, col_idx]
            scores = []
            bar_colors = []
            bar_labels = []

            for model_name in models:
                row = df_layer[df_layer['model'] == model_name]
                if len(row) > 0:
                    scores.append(row[m_key].values[0])
                else:
                    scores.append(0.0)
                bar_colors.append(colors_dict.get(model_name, '#555555'))
                short_label = model_name.replace("RootSplit_LGBM", "LGB(Root)").replace("Feature129_LGBM", "LGB(129)").replace("YHead_MLP", "MLP(YHead)").replace("SingleHead129_MLP", "MLP(129)")
                bar_labels.append(short_label)

            x_pos = np.arange(len(models))
            bars = ax.bar(x_pos, scores, color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.7)

            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.annotate(f"{h:.3f}",
                                xy=(bar.get_x() + bar.get_width() / 2, h),
                                xytext=(0, 2),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=7.5, fontweight='bold')

            if row_idx == 0:
                ax.set_title(metrics_titles[m_key], fontsize=11, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(f"Layer {layer}\nScore", fontsize=11, fontweight='bold')

            ax.set_xticks(x_pos)
            ax.set_xticklabels(bar_labels, rotation=25, fontsize=8)
            ax.set_ylim([0, 1.1])
            ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    combined_comp_path = os.path.join(output_dir, "model_comparison_combined.png")
    plt.savefig(combined_comp_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  └─ 儲存 4x6 跨層數總覽條形圖: {combined_comp_path}")

    print("\n" + "=" * 80)
    print(f"所有圖表已成功生成，獨立儲存於: {output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()
