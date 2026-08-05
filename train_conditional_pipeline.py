"""
模型內建條件分流分類器 (RootSplit-LGBM & YHead-MLP) 全流程訓練與評估腳本
======================================================================
針對 6 個 Hidden State 特徵層，訓練以 y1 (模型回覆安全性) 為條件預測 y2 (提示詞有害性) 之條件模型，
並於驗證集 (2,000 筆) 與獨立評估集 experiment_results_eval (2,210 筆) 上
評估整體與 Head 0 (y1=0) / Head 1 (y1=1) 之 AUC, Acc, Brier 指標。
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss

from conditional_models import RootSplitLGBMClassifier, YHeadMLPPyTorchClassifier

def extract_y1_y2(df):
    y1 = df['model_reply'].str.lower().str.contains('unsafe').astype(int).values if 'model_reply' in df.columns else df['y1'].values
    y2 = df['data_type'].str.contains('harmful').astype(int).values if 'data_type' in df.columns else df['y2'].values
    return y1, y2

def main():
    print("=" * 75)
    print("開始執行模型內建條件分流分類器 (Conditional Branching Pipeline) 訓練與評估")
    print("=" * 75)

    train_path = "data/experiment_results_train_10000.pkl"
    eval_path = "data/experiment_results_eval.pkl"
    output_dir = "results/conditional_training"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[1] 載入主訓練數據集: {train_path} ...")
    df_train = pd.read_pickle(train_path)
    X_train_3d = np.array(df_train['hidden_state'].tolist())
    y1_train, y2_train = extract_y1_y2(df_train)
    print(f"  └─ 成功載入 {len(df_train)} 筆訓練樣本 | y1 (Unsafe) 比例: {np.mean(y1_train):.2%} | y2 (Harmful) 比例: {np.mean(y2_train):.2%}")

    print(f"\n[2] 載入獨立評估數據集: {eval_path} ...")
    df_eval = pd.read_pickle(eval_path)
    X_eval_3d = np.array(df_eval['hidden_state'].tolist())
    y1_eval, y2_eval = extract_y1_y2(df_eval)
    print(f"  └─ Eval 評估集: {len(df_eval)} 筆 | y1 (Unsafe) 比例: {np.mean(y1_eval):.2%}")

    # 劃分 80% 訓練 (8,000) / 20% 驗證 (2,000)
    train_idx, val_idx = train_test_split(np.arange(len(df_train)), test_size=0.2, random_state=42, stratify=y2_train)

    summary_records = []

    for layer in range(1, 7):
        layer_idx = layer - 1
        print("\n" + "=" * 65)
        print(f" 正在處理特徵層: 【LAYER {layer} / 6】")
        print("=" * 65)

        layer_dir = os.path.join(output_dir, f"layer_{layer}")
        os.makedirs(layer_dir, exist_ok=True)

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
        idx_res, y2_tr_bal = rus.fit_resample(np.arange(len(X_tr_scaled)).reshape(-1, 1), y2_tr)
        idx_res = idx_res.flatten()
        
        X_tr_bal = X_tr_scaled[idx_res]
        y1_tr_bal = y1_tr[idx_res]

        pca = PCA(n_components=128, random_state=42)
        X_tr_pca = pca.fit_transform(X_tr_bal)
        X_val_pca = pca.transform(X_val_scaled)
        X_ev_pca = pca.transform(X_ev_scaled)

        # 3.1 RootSplitLGBMClassifier
        print(f"  [Layer {layer}] 訓練 RootSplit-LGBM 模型...")
        lgb_model = RootSplitLGBMClassifier(max_depth=4, num_leaves=15, n_estimators=100, learning_rate=0.05, random_state=42)
        lgb_model.fit(X_tr_pca, y1_tr_bal, y2_tr_bal)
        joblib.dump(lgb_model, os.path.join(layer_dir, "rootsplit_lgbm.joblib"))

        # 3.2 YHeadMLPPyTorchClassifier
        print(f"  [Layer {layer}] 訓練 YHead-MLP PyTorch 模型...")
        mlp_model = YHeadMLPPyTorchClassifier(input_dim=128, epochs=40, batch_size=64, lr=1e-3, random_state=42)
        mlp_model.fit(X_tr_pca, y1_tr_bal, y2_tr_bal, verbose=False)
        joblib.dump(mlp_model, os.path.join(layer_dir, "yhead_mlp.joblib"))

        eval_datasets = [
            ("Val_Set", X_val_pca, y1_val, y2_val),
            ("Eval_Set", X_ev_pca, y1_eval, y2_eval)
        ]

        for split_name, X_split, y1_split, y2_split in eval_datasets:
            for model_name, model in [("RootSplit_LGBM", lgb_model), ("YHead_MLP", mlp_model)]:
                p_cond = model.predict_proba(X_split, y1_split)[:, 1]
                p_head0 = model.predict_proba_head0(X_split)[:, 1]
                p_head1 = model.predict_proba_head1(X_split)[:, 1]

                acc_cond = accuracy_score(y2_split, (p_cond >= 0.5).astype(int))
                auc_cond = roc_auc_score(y2_split, p_cond)
                brier_cond = brier_score_loss(y2_split, p_cond)

                # Head 0 (y1=0)
                mask0 = (y1_split == 0)
                if np.sum(mask0) > 0 and len(np.unique(y2_split[mask0])) > 1:
                    acc_h0 = accuracy_score(y2_split[mask0], (p_head0[mask0] >= 0.5).astype(int))
                    auc_h0 = roc_auc_score(y2_split[mask0], p_head0[mask0])
                    brier_h0 = brier_score_loss(y2_split[mask0], p_head0[mask0])
                else:
                    acc_h0 = accuracy_score(y2_split[mask0], (p_head0[mask0] >= 0.5).astype(int)) if np.sum(mask0) > 0 else 0.0
                    auc_h0 = 0.5
                    brier_h0 = brier_score_loss(y2_split[mask0], p_head0[mask0]) if np.sum(mask0) > 0 else 0.0

                # Head 1 (y1=1)
                mask1 = (y1_split == 1)
                if np.sum(mask1) > 0 and len(np.unique(y2_split[mask1])) > 1:
                    acc_h1 = accuracy_score(y2_split[mask1], (p_head1[mask1] >= 0.5).astype(int))
                    auc_h1 = roc_auc_score(y2_split[mask1], p_head1[mask1])
                    brier_h1 = brier_score_loss(y2_split[mask1], p_head1[mask1])
                else:
                    acc_h1 = accuracy_score(y2_split[mask1], (p_head1[mask1] >= 0.5).astype(int)) if np.sum(mask1) > 0 else 0.0
                    auc_h1 = 0.5
                    brier_h1 = brier_score_loss(y2_split[mask1], p_head1[mask1]) if np.sum(mask1) > 0 else 0.0

                summary_records.append({
                    "layer": layer,
                    "model": model_name,
                    "dataset": split_name,
                    "acc_cond": acc_cond,
                    "auc_cond": auc_cond,
                    "brier_cond": brier_cond,
                    "acc_head0": acc_h0,
                    "auc_head0": auc_h0,
                    "brier_head0": brier_h0,
                    "acc_head1": acc_h1,
                    "auc_head1": auc_h1,
                    "brier_head1": brier_h1,
                })

                print(f"     [{split_name:8s} | {model_name:14s}] Cond AUC: {auc_cond:.4f} (Acc: {acc_cond:.4f}) | Head0 AUC: {auc_h0:.4f} | Head1 AUC: {auc_h1:.4f}")

    df_res = pd.DataFrame(summary_records)
    csv_out = os.path.join(output_dir, "conditional_models_evaluation_summary.csv")
    df_res.to_csv(csv_out, index=False, encoding='utf-8-sig')
    print("\n" + "=" * 75)
    print(f"訓練與評估完成！結果已儲存至: {csv_out}")
    print("=" * 75)

if __name__ == "__main__":
    main()
