"""
8月6日 LLM 隱藏狀態機率校正與元評估框架 — 獨立訓練與評估腳本
======================================================================
針對 Layer 6 訓練 6 種模型策略：
1. LR_Hard_Dual
2. LR_Interaction
3. LGB_Hard_Dual
4. RootSplit_LGBM
5. MLP_Hard_Dual
6. YHead_MLP

資料集：v1_train_full.pkl (Train) / v1_val.pkl (Val)
結果儲存至: results/v2_framework/framework_training/
"""

import os
import sys
# Add current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import joblib
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
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

from conditional_models import (
    RootSplitLGBMClassifier,
    YHeadMLPPyTorchClassifier,
    SingleHeadMLPPyTorchClassifier,
    HardDualClassifierWrapper,
    LRInteractionClassifier
)

def extract_y1_y2(df):
    y1 = df['model_reply'].str.lower().str.contains('unsafe').astype(int).values if 'model_reply' in df.columns else df['y1'].values
    y2 = df['data_type'].str.contains('harmful').astype(int).values if 'data_type' in df.columns else df['y2'].values
    return y1, y2

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    print("=" * 80)
    print("V2 隱藏狀態機率校正框架 — 開始訓練 Layer 6 × 6模型 (全量特徵)")
    print("=" * 80)

    train_path = "data/v1_train_full.pkl"
    val_path = "data/v1_val.pkl"
    output_dir = "outputs/v2_framework/framework_training"
    os.makedirs(output_dir, exist_ok=True)
    pca_status = "full_1024d"

    print(f"\n[1] 載入訓練集: {train_path} ...")
    df_train = pd.read_pickle(train_path)
    X_3d_train = np.array(df_train['hidden_state'].tolist())
    y1_tr, y2_tr = extract_y1_y2(df_train)
    print(f"  └─ 成功載入 {len(df_train)} 筆資料 | y1 (Unsafe): {np.mean(y1_tr):.2%} | y2 (Harmful): {np.mean(y2_tr):.2%}")

    print(f"\n[2] 載入驗證集: {val_path} ...")
    df_val = pd.read_pickle(val_path)
    X_3d_val = np.array(df_val['hidden_state'].tolist())
    y1_val, y2_val = extract_y1_y2(df_val)
    val_idx = np.arange(len(df_val))
    print(f"  └─ 成功載入 {len(df_val)} 筆資料 | y1 (Unsafe): {np.mean(y1_val):.2%} | y2 (Harmful): {np.mean(y2_val):.2%}")

    summary_records = []
    val_predictions = {
        'val_idx': val_idx,
        'y1_val': y1_val,
        'y2_val': y2_val,
        'layers': {}
    }

    layers_to_train = [6]

    for layer in layers_to_train:
        layer_idx = layer - 1
        print("\n" + "=" * 70)
        print(f"【Layer {layer}】開始訓練與評估 (無 PCA) | PCA: {pca_status}")
        print("=" * 70)

        layer_dir = os.path.join(output_dir, f"layer_{layer}")
        os.makedirs(layer_dir, exist_ok=True)

        X_tr = X_3d_train[:, layer_idx, :]
        X_val = X_3d_val[:, layer_idx, :]

        # 標準化
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)

        input_dim = X_tr_scaled.shape[1]

        # 6 種模型實體定義
        models_dict = {
            "LR_Hard_Dual": HardDualClassifierWrapper(LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver='liblinear')),
            "LR_Interaction": LRInteractionClassifier(C=1.0, max_iter=1000, random_state=42),
            "LGB_Hard_Dual": HardDualClassifierWrapper(lgb.LGBMClassifier(max_depth=-1, num_leaves=31, n_estimators=100, learning_rate=0.05, random_state=42, n_jobs=1, verbose=-1)),
            "RootSplit_LGBM": RootSplitLGBMClassifier(max_depth=-1, num_leaves=31, n_estimators=100, learning_rate=0.05, reg_alpha=0.0, reg_lambda=0.0, random_state=42),
            "MLP_Hard_Dual": HardDualClassifierWrapper(SingleHeadMLPPyTorchClassifier(input_dim=input_dim, epochs=15, lr=1e-3, batch_size=64, random_state=42)),
            "YHead_MLP": YHeadMLPPyTorchClassifier(input_dim=input_dim, epochs=15, lr=1e-3, batch_size=64, random_state=42)
        }

        val_predictions['layers'][layer] = {}

        for model_name, model in models_dict.items():
            print(f"  ├─ [Layer {layer}] 正在訓練: {model_name:20s} ...")
            try:
                model.fit(X_tr_scaled, y1_tr, y2_tr)
                joblib.dump(model, os.path.join(layer_dir, f"{model_name.lower()}.joblib"))

                # 預測 Val_Set
                proba = model.predict_proba(X_val_scaled, y1_val)[:, 1]
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
    print(f"6 個模型訓練與 Val_Set 評估完成！ ({pca_status})")
    print(f"  ├─ 數據摘要 CSV: {csv_out}")
    print(f"  └─ 驗證集預測檔: {preds_out}")
    print("=" * 80)

if __name__ == "__main__":
    main()
