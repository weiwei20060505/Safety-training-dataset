"""
8月6日 LLM 隱藏狀態機率校正與元評估框架 — 獨立訓練與評估腳本
======================================================================
針對 Layer 3, 4, 5, 6 訓練 4 種模型策略 (共 16 個模型)：
1. RootSplit_LGBM
2. FeaturePlusY1_LGBM
3. YHead_MLP
4. SingleHead_MLP

資料集切分：60% Train (6,000筆) / 20% Val (2,000筆) / 20% Test (2,000筆)
結果儲存至: results/v2_framework/framework_training/{with_pca,without_pca}/
"""

import os
import sys
# Add current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)

from conditional_models import (
    RootSplitLGBMClassifier,
    FeaturePlusY1LGBMClassifier,
    YHeadMLPPyTorchClassifier,
    SingleHeadMLPPyTorchClassifier
)

def extract_y1_y2(df):
    y1 = df['model_reply'].str.lower().str.contains('unsafe').astype(int).values if 'model_reply' in df.columns else df['y1'].values
    y2 = df['data_type'].str.contains('harmful').astype(int).values if 'data_type' in df.columns else df['y2'].values
    return y1, y2

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    print("=" * 80)
    print("8月6日 LLM 隱藏狀態機率校正框架 — 開始訓練 16 個模型 (Layer 3~6 × 4模型)")
    print("=" * 80)

    train_path = "data/experiment_results_train_10000.pkl"
    output_dir = "results/v2_framework/framework_training"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[1] 載入主數據集: {train_path} ...")
    df = pd.read_pickle(train_path)
    X_3d = np.array(df['hidden_state'].tolist())
    y1_all, y2_all = extract_y1_y2(df)
    print(f"  └─ 成功載入 {len(df)} 筆數據 | y1 (Unsafe): {np.mean(y1_all):.2%} | y2 (Harmful): {np.mean(y2_all):.2%}")

    # 切分: 60% Train, 20% Val, 20% Test
    train_val_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=42, stratify=y2_all
    )
    y2_train_val = y2_all[train_val_idx]

    train_idx_sub, val_idx_sub = train_test_split(
        np.arange(len(train_val_idx)), test_size=0.25, random_state=42, stratify=y2_train_val
    )

    train_idx = train_val_idx[train_idx_sub] # 6,000 筆
    val_idx = train_val_idx[val_idx_sub]     # 2,000 筆

    print(f"  └─ 資料分割: Train = {len(train_idx)} 筆 (60%) | Val = {len(val_idx)} 筆 (20%) | Test = {len(test_idx)} 筆 (20%)")

    summary_records = []
    val_predictions = {
        'val_idx': val_idx,
        'y1_val': y1_all[val_idx],
        'y2_val': y2_all[val_idx],
        'layers': {}
    }

    layers_to_train = [3, 4, 5, 6]

    for layer in layers_to_train:
        layer_idx = layer - 1
        print("\n" + "=" * 70)
        print(f"【Layer {layer} / 6】開始訓練與評估 (無 RandomUnderSampler) | PCA: {pca_status}")
        print("=" * 70)

        layer_dir = os.path.join(output_dir, f"layer_{layer}")
        os.makedirs(layer_dir, exist_ok=True)

        X_tr = X_3d[train_idx, layer_idx, :]
        y1_tr, y2_tr = y1_all[train_idx], y2_all[train_idx]

        X_val = X_3d[val_idx, layer_idx, :]
        y1_val, y2_val = y1_all[val_idx], y2_all[val_idx]

        # 標準化
        scaler = StandardScaler()
        X_tr_pca = scaler.fit_transform(X_tr)
        X_val_pca = scaler.transform(X_val)

        input_dim = X_tr_pca.shape[1]

        # 4 種模型實體定義
        models_dict = {
            "RootSplit_LGBM": RootSplitLGBMClassifier(max_depth=-1, num_leaves=31, n_estimators=100, learning_rate=0.05, reg_alpha=0.0, reg_lambda=0.0, random_state=42),
            "FeaturePlusY1_LGBM": FeaturePlusY1LGBMClassifier(max_depth=-1, num_leaves=31, n_estimators=100, learning_rate=0.05, reg_alpha=0.0, reg_lambda=0.0, random_state=42),
            "YHead_MLP": YHeadMLPPyTorchClassifier(input_dim=input_dim, epochs=15, lr=1e-3, batch_size=64, random_state=42),
            "SingleHead_MLP": SingleHeadMLPPyTorchClassifier(input_dim=input_dim, epochs=15, lr=1e-3, batch_size=64, random_state=42)
        }

        val_predictions['layers'][layer] = {}

        for model_name, model in models_dict.items():
            print(f"  ├─ [Layer {layer}] 正在訓練: {model_name:20s} ...")
            try:
                model.fit(X_tr_pca, y1_tr, y2_tr)
                joblib.dump(model, os.path.join(layer_dir, f"{model_name.lower()}.joblib"))

                # 預測 Val_Set
                proba = model.predict_proba(X_val_pca, y1_val)[:, 1]
                pred = (proba >= 0.5).astype(int)

                # 計算 6 大指標
                acc = accuracy_score(y2_val, pred)
                bal_acc = balanced_accuracy_score(y2_val, pred)
                prec = precision_score(y2_val, pred, zero_division=0)
                rec = recall_score(y2_val, pred, zero_division=0)
                f1 = f1_score(y2_val, pred, zero_division=0)
                auc_val = roc_auc_score(y2_val, proba)

                # 計算 ROC 曲線 (fpr, tpr)
                fpr, tpr, _ = roc_curve(y2_val, proba)

                summary_records.append({
                    "layer": layer,
                    "model": model_name,
                    "dataset": "Val_Set",
                    "accuracy": acc,
                    "balanced_accuracy": bal_acc,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "auc": auc_val
                })

                val_predictions['layers'][layer][model_name] = {
                    'proba': proba,
                    'pred': pred,
                    'fpr': fpr,
                    'tpr': tpr,
                    'auc': auc_val
                }

                print(f"  │    └─ Acc: {acc:.4f} | BalAcc: {bal_acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc_val:.4f}")

            except Exception as e:
                print(f"  │    └─ [Error] 訓練或評估失敗: {e}")
                import traceback
                traceback.print_exc()

    # 儲存 Summary CSV 與 Val 預測結果 joblib
    df_summary = pd.DataFrame(summary_records)
    csv_out = os.path.join(output_dir, "framework_evaluation_summary.csv")
    df_summary.to_csv(csv_out, index=False, encoding='utf-8-sig')

    preds_out = os.path.join(output_dir, "val_predictions.joblib")
    joblib.dump(val_predictions, preds_out)

    print("\n" + "=" * 80)
    print(f"16 個模型訓練與 Val_Set 評估完成！ ({pca_status})")
    print(f"  ├─ 數據摘要 CSV: {csv_out}")
    print(f"  └─ 驗證集預測檔: {preds_out}")
    print("=" * 80)

if __name__ == "__main__":
    main()
