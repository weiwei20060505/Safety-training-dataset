"""
模型內建條件分流分類器 (RootSplit-LGBM & YHead-MLP) ROC / AUC 曲線繪製腳本
======================================================================
針對 6 個 Hidden State 特徵層，繪製兩大條件模型在 Validation Set 與 Eval Set 上之
整體條件 AUC 及 Head 0 (Safe) / Head 1 (Unsafe) 分支 ROC 曲線圖，
並將結果儲存於獨立資料夾 `results/plots_conditional/`
"""

import os
import sys
# Add current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import roc_curve, auc

from conditional_models import RootSplitLGBMClassifier, YHeadMLPPyTorchClassifier

# 設定字體與風格
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def extract_y1_y2(df):
    y1 = df['model_reply'].str.lower().str.contains('unsafe').astype(int).values if 'model_reply' in df.columns else df['y1'].values
    y2 = df['data_type'].str.contains('harmful').astype(int).values if 'data_type' in df.columns else df['y2'].values
    return y1, y2

def main():
    print("=" * 75)
    print("開始繪製條件分流模型 (RootSplit-LGBM vs YHead-MLP) 之 ROC / AUC 曲線")
    print("=" * 75)

    train_path = "data/experiment_results_train_10000.pkl"
    eval_path = "data/experiment_results_eval.pkl"
    model_dir = "results/v2_framework/conditional_training"
    
    # 獨立的輸出資料夾，與原本 results/plots 區分開來
    output_dir = "results/v2_framework/plots_conditional"
    roc_dir = os.path.join(output_dir, "08_ROC_Curves")
    by_layer_dir = os.path.join(output_dir, "by_layer")
    os.makedirs(roc_dir, exist_ok=True)
    os.makedirs(by_layer_dir, exist_ok=True)

    print(f"\n[1] 載入訓練與評估數據集...")
    df_train = pd.read_pickle(train_path)
    X_train_3d = np.array(df_train['hidden_state'].tolist())
    y1_train, y2_train = extract_y1_y2(df_train)

    df_eval = pd.read_pickle(eval_path)
    X_eval_3d = np.array(df_eval['hidden_state'].tolist())
    y1_eval, y2_eval = extract_y1_y2(df_eval)

    train_idx, val_idx = train_test_split(np.arange(len(df_train)), test_size=0.2, random_state=42, stratify=y2_train)

    # 儲存所有層的預測曲線資料
    roc_data = {
        'Val_Set': {layer: {} for layer in range(1, 7)},
        'Eval_Set': {layer: {} for layer in range(1, 7)}
    }

    for layer in range(1, 7):
        layer_idx = layer - 1
        print(f"  └─ 處理 Layer {layer} 特徵與載入模型...")

        X_tr = X_train_3d[train_idx, layer_idx, :]
        y1_tr, y2_tr = y1_train[train_idx], y2_train[train_idx]

        X_val = X_train_3d[val_idx, layer_idx, :]
        y1_val, y2_val = y1_train[val_idx], y2_train[val_idx]

        X_ev = X_eval_3d[:, layer_idx, :]

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)
        X_ev_scaled = scaler.transform(X_ev)

        rus = RandomUnderSampler(sampling_strategy=1.0, random_state=42)
        idx_res, _ = rus.fit_resample(np.arange(len(X_tr_scaled)).reshape(-1, 1), y2_tr)
        idx_res = idx_res.flatten()

        # 載入模型
        lgb_path = os.path.join(model_dir, f"layer_{layer}", "rootsplit_lgbm.joblib")
        mlp_path = os.path.join(model_dir, f"layer_{layer}", "yhead_mlp.joblib")

        lgb_model = joblib.load(lgb_path)
        mlp_model = joblib.load(mlp_path)

        datasets = [
            ("Val_Set", X_val_scaled, y1_val, y2_val),
            ("Eval_Set", X_ev_scaled, y1_eval, y2_eval)
        ]

        for split_name, X_split, y1_split, y2_split in datasets:
            for model_name, model in [("RootSplit_LGBM", lgb_model), ("YHead_MLP", mlp_model)]:
                p_cond = model.predict_proba(X_split, y1_split)[:, 1]
                p_h0 = model.predict_proba_head0(X_split)[:, 1]
                p_h1 = model.predict_proba_head1(X_split)[:, 1]

                # Overall Cond ROC
                fpr_cond, tpr_cond, _ = roc_curve(y2_split, p_cond)
                auc_cond = auc(fpr_cond, tpr_cond)

                # Head 0 ROC (y1=0)
                mask0 = (y1_split == 0)
                if np.sum(mask0) > 0 and len(np.unique(y2_split[mask0])) > 1:
                    fpr_h0, tpr_h0, _ = roc_curve(y2_split[mask0], p_h0[mask0])
                    auc_h0 = auc(fpr_h0, tpr_h0)
                else:
                    fpr_h0, tpr_h0, auc_h0 = [0, 1], [0, 1], 0.5

                # Head 1 ROC (y1=1)
                mask1 = (y1_split == 1)
                if np.sum(mask1) > 0 and len(np.unique(y2_split[mask1])) > 1:
                    fpr_h1, tpr_h1, _ = roc_curve(y2_split[mask1], p_h1[mask1])
                    auc_h1 = auc(fpr_h1, tpr_h1)
                else:
                    fpr_h1, tpr_h1, auc_h1 = [0, 1], [0, 1], 0.5

                roc_data[split_name][layer][model_name] = {
                    'cond': (fpr_cond, tpr_cond, auc_cond),
                    'head0': (fpr_h0, tpr_h0, auc_h0),
                    'head1': (fpr_h1, tpr_h1, auc_h1)
                }

    print("\n[2] 開始繪製 1x6 多圖對比 ROC 曲線...")

    # 顏色配置
    colors_lgbm = '#D9381E'  # Crimson Red
    colors_mlp = '#2B547E'   # Deep Blue

    for split_name in ['Val_Set', 'Eval_Set']:
        # 圖 1: 1x6 各層 RootSplit-LGBM vs YHead-MLP (整體 Conditional ROC)
        fig, axes = plt.subplots(1, 6, figsize=(24, 4.5), sharey=True)
        fig.suptitle(f"Conditional Models Overall ROC Curves Across Layers ({split_name})", fontsize=16, fontweight='bold', y=1.02)

        for layer in range(1, 7):
            ax = axes[layer - 1]
            data_lgb = roc_data[split_name][layer]['RootSplit_LGBM']['cond']
            data_mlp = roc_data[split_name][layer]['YHead_MLP']['cond']

            ax.plot(data_lgb[0], data_lgb[1], color=colors_lgbm, lw=2, label=f"RootSplit-LGBM (AUC={data_lgb[2]:.4f})")
            ax.plot(data_mlp[0], data_mlp[1], color=colors_mlp, lw=2, linestyle='--', label=f"YHead-MLP (AUC={data_mlp[2]:.4f})")
            ax.plot([0, 1], [0, 1], color='gray', linestyle=':', lw=1.2, label='Chance')

            ax.set_title(f"Layer {layer}", fontsize=13, fontweight='bold')
            ax.set_xlabel("False Positive Rate", fontsize=11)
            if layer == 1:
                ax.set_ylabel("True Positive Rate", fontsize=11)
            ax.legend(loc='lower right', fontsize=9, frameon=True)
            ax.set_xlim([-0.02, 1.02])
            ax.set_ylim([-0.02, 1.02])
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(roc_dir, f"conditional_overall_roc_1x6_{split_name.lower()}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ├─ 儲存 1x6 Overall ROC: {save_path}")

        # 圖 2: 2x6 分支 Head 0 (y1=0) 與 Head 1 (y1=1) ROC 比較圖
        fig, axes = plt.subplots(2, 6, figsize=(24, 8.5), sharex=True, sharey=True)
        fig.suptitle(f"Conditional Models Branch Head ROC Curves Across Layers ({split_name})", fontsize=16, fontweight='bold', y=0.98)

        for layer in range(1, 7):
            # Row 0: Head 0 (y1=0, Safe Branch)
            ax0 = axes[0, layer - 1]
            data_lgb_h0 = roc_data[split_name][layer]['RootSplit_LGBM']['head0']
            data_mlp_h0 = roc_data[split_name][layer]['YHead_MLP']['head0']

            ax0.plot(data_lgb_h0[0], data_lgb_h0[1], color=colors_lgbm, lw=2, label=f"LGBM Head0 (AUC={data_lgb_h0[2]:.4f})")
            ax0.plot(data_mlp_h0[0], data_mlp_h0[1], color=colors_mlp, lw=2, linestyle='--', label=f"MLP Head0 (AUC={data_mlp_h0[2]:.4f})")
            ax0.plot([0, 1], [0, 1], color='gray', linestyle=':', lw=1.2)
            ax0.set_title(f"Layer {layer} - Head 0 (y1=0)", fontsize=11, fontweight='bold')
            ax0.legend(loc='lower right', fontsize=8.5, frameon=True)
            ax0.grid(True, alpha=0.3)
            if layer == 1:
                ax0.set_ylabel("Head 0 (Safe)\nTrue Positive Rate", fontsize=11)

            # Row 1: Head 1 (y1=1, Unsafe Branch)
            ax1 = axes[1, layer - 1]
            data_lgb_h1 = roc_data[split_name][layer]['RootSplit_LGBM']['head1']
            data_mlp_h1 = roc_data[split_name][layer]['YHead_MLP']['head1']

            ax1.plot(data_lgb_h1[0], data_lgb_h1[1], color='#E67E22', lw=2, label=f"LGBM Head1 (AUC={data_lgb_h1[2]:.4f})")
            ax1.plot(data_mlp_h1[0], data_mlp_h1[1], color='#8E44AD', lw=2, linestyle='--', label=f"MLP Head1 (AUC={data_mlp_h1[2]:.4f})")
            ax1.plot([0, 1], [0, 1], color='gray', linestyle=':', lw=1.2)
            ax1.set_title(f"Layer {layer} - Head 1 (y1=1)", fontsize=11, fontweight='bold')
            ax1.set_xlabel("False Positive Rate", fontsize=11)
            ax1.legend(loc='lower right', fontsize=8.5, frameon=True)
            ax1.grid(True, alpha=0.3)
            if layer == 1:
                ax1.set_ylabel("Head 1 (Unsafe)\nTrue Positive Rate", fontsize=11)

        plt.tight_layout()
        save_path_branches = os.path.join(roc_dir, f"conditional_branches_roc_2x6_{split_name.lower()}.png")
        plt.savefig(save_path_branches, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ├─ 儲存 2x6 Branches ROC: {save_path_branches}")

    # 圖 3: 依層數拆分的獨立單圖 (Per Layer Detailed Plots)
    for layer in range(1, 7):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        fig.suptitle(f"Layer {layer} Conditional Classification ROC Curves", fontsize=14, fontweight='bold')

        for idx, split_name in enumerate(['Val_Set', 'Eval_Set']):
            ax = axes[idx]
            d_lgb_cond = roc_data[split_name][layer]['RootSplit_LGBM']['cond']
            d_mlp_cond = roc_data[split_name][layer]['YHead_MLP']['cond']
            d_lgb_h0 = roc_data[split_name][layer]['RootSplit_LGBM']['head0']
            d_mlp_h0 = roc_data[split_name][layer]['YHead_MLP']['head0']
            d_lgb_h1 = roc_data[split_name][layer]['RootSplit_LGBM']['head1']
            d_mlp_h1 = roc_data[split_name][layer]['YHead_MLP']['head1']

            ax.plot(d_lgb_cond[0], d_lgb_cond[1], color=colors_lgbm, lw=2, label=f"LGBM Cond (AUC={d_lgb_cond[2]:.4f})")
            ax.plot(d_mlp_cond[0], d_mlp_cond[1], color=colors_mlp, lw=2, label=f"MLP Cond (AUC={d_mlp_cond[2]:.4f})")
            ax.plot(d_lgb_h0[0], d_lgb_h0[1], color='#27AE60', lw=1.5, linestyle=':', label=f"LGBM Head0 (AUC={d_lgb_h0[2]:.4f})")
            ax.plot(d_mlp_h0[0], d_mlp_h0[1], color='#16A085', lw=1.5, linestyle='--', label=f"MLP Head0 (AUC={d_mlp_h0[2]:.4f})")
            ax.plot(d_lgb_h1[0], d_lgb_h1[1], color='#E67E22', lw=1.5, linestyle=':', label=f"LGBM Head1 (AUC={d_lgb_h1[2]:.4f})")
            ax.plot(d_mlp_h1[0], d_mlp_h1[1], color='#8E44AD', lw=1.5, linestyle='--', label=f"MLP Head1 (AUC={d_mlp_h1[2]:.4f})")
            ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)

            ax.set_title(f"{split_name}", fontsize=12, fontweight='bold')
            ax.set_xlabel("False Positive Rate", fontsize=11)
            ax.set_ylabel("True Positive Rate", fontsize=11)
            ax.legend(loc='lower right', fontsize=8.5, frameon=True)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        layer_save_path = os.path.join(by_layer_dir, f"layer_{layer}_roc_curves.png")
        plt.savefig(layer_save_path, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"\n" + "=" * 75)
    print(f"所有 ROC 曲線圖已成功生成並獨立儲存至: {output_dir}")
    print("=" * 75)

if __name__ == "__main__":
    main()
