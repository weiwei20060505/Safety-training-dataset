"""
8月6日 LLM 隱藏狀態機率校正與元評估框架 — 階段二：獨立校準與評估腳本
======================================================================
針對 Layer 3, 4, 5, 6 訓練好的 16 個模型，在 3 個獨立測試集（Test 1, Test 2, Eval）上
執行 PAVA (Isotonic Regression) 保序回歸校準與子群 ($y_1$) 分流校準。

算出的 Raw Score, Calibrated Probability, Brier Components 等數據完整儲存至:
  results/framework_calibration/
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression

from utils_calibration import calculate_all_metrics, brier_score_decomposition
from conditional_models import (
    RootSplitLGBMClassifier,
    Feature129LGBMClassifier,
    YHeadMLPPyTorchClassifier,
    SingleHead129MLPPyTorchClassifier
)

def extract_y1_y2(df):
    y1 = df['model_reply'].str.lower().str.contains('unsafe').astype(int).values if 'model_reply' in df.columns else df['y1'].values
    y2 = df['data_type'].str.contains('harmful').astype(int).values if 'data_type' in df.columns else df['y2'].values
    return y1, y2

def main():
    print("=" * 80)
    print("8月6日 LLM 隱藏狀態機率校正框架 — 開始執行階段二：子群機率校準 (PAVA)")
    print("=" * 80)

    train_path = "data/experiment_results_train_10000.pkl"
    model_dir = "results/framework_training"
    output_dir = "results/framework_calibration"
    os.makedirs(output_dir, exist_ok=True)

    # 1. 讀取基準 10,000 訓練/驗證集以擬合 Scaler, PCA 與 Isotonic Calibration
    print(f"\n[1] 載入基準 Train/Val 數據集: {train_path} ...")
    df_train = pd.read_pickle(train_path)
    X_train_3d = np.array(df_train['hidden_state'].tolist())
    y1_train_all, y2_train_all = extract_y1_y2(df_train)

    train_val_idx, test_idx = train_test_split(
        np.arange(len(df_train)), test_size=0.2, random_state=42, stratify=y2_train_all
    )
    y2_train_val = y2_train_all[train_val_idx]

    train_idx_sub, val_idx_sub = train_test_split(
        np.arange(len(train_val_idx)), test_size=0.25, random_state=42, stratify=y2_train_val
    )

    train_idx = train_val_idx[train_idx_sub] # 6,000 筆
    val_idx = train_val_idx[val_idx_sub]     # 2,000 筆

    # 2. 載入 3 個獨立測試集 (Test 1, Test 2, Eval)
    test_datasets = {}
    
    test1_path = "data/test1.pkl"
    if os.path.exists(test1_path):
        t1_obj = joblib.load(test1_path)
        df_t1 = t1_obj['y2'] if isinstance(t1_obj, dict) and 'y2' in t1_obj else t1_obj
        y1_t1, y2_t1 = extract_y1_y2(df_t1)
        test_datasets['test1'] = {
            'df': df_t1,
            'X_3d': np.array(df_t1['hidden_state'].tolist()),
            'y1': y1_t1,
            'y2': y2_t1
        }
        print(f"  ├─ 載入 Test 1 (`test1.pkl`): {len(df_t1)} 筆")

    test2_path = "data/test2.pkl"
    if os.path.exists(test2_path):
        t2_obj = joblib.load(test2_path)
        df_t2 = t2_obj['y2'] if isinstance(t2_obj, dict) and 'y2' in t2_obj else t2_obj
        y1_t2, y2_t2 = extract_y1_y2(df_t2)
        test_datasets['test2'] = {
            'df': df_t2,
            'X_3d': np.array(df_t2['hidden_state'].tolist()),
            'y1': y1_t2,
            'y2': y2_t2
        }
        print(f"  ├─ 載入 Test 2 (`test2.pkl`): {len(df_t2)} 筆")

    eval_path = "data/experiment_results_eval.pkl"
    if os.path.exists(eval_path):
        df_ev = pd.read_pickle(eval_path)
        y1_ev, y2_ev = extract_y1_y2(df_ev)
        test_datasets['eval'] = {
            'df': df_ev,
            'X_3d': np.array(df_ev['hidden_state'].tolist()),
            'y1': y1_ev,
            'y2': y2_ev
        }
        print(f"  └─ 載入 Eval / Test 3 (`experiment_results_eval.pkl`): {len(df_ev)} 筆")

    layers = [3, 4, 5, 6]
    models = ["RootSplit_LGBM", "Feature129_LGBM", "YHead_MLP", "SingleHead129_MLP"]

    calibration_records = []
    calibration_data = {
        'layers': {}
    }

    for layer in layers:
        layer_idx = layer - 1
        print("\n" + "=" * 75)
        print(f"【Layer {layer} / 6】進行模型加載、分數預測與 PAVA 機率校準")
        print("=" * 75)

        X_tr = X_train_3d[train_idx, layer_idx, :]
        y1_tr, y2_tr = y1_train_all[train_idx], y2_train_all[train_idx]

        X_val = X_train_3d[val_idx, layer_idx, :]
        y1_val, y2_val = y1_train_all[val_idx], y2_train_all[val_idx]

        # 擬合 Scaler & PCA
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)

        pca = PCA(n_components=128, random_state=42)
        X_tr_pca = pca.fit_transform(X_tr_scaled)
        X_val_pca = pca.transform(X_val_scaled)

        calibration_data['layers'][layer] = {}

        for model_name in models:
            model_path = os.path.join(model_dir, f"layer_{layer}", f"{model_name.lower()}.joblib")
            if not os.path.exists(model_path):
                print(f"  ⚠️ 跳過 {model_name} (找不到檔案: {model_path})")
                continue

            model = joblib.load(model_path)
            print(f"  ├─ [Layer {layer}] 載入模型: {model_name:20s}")

            # [Option A] 使用 Test 1 數據集的預測分數擬合 PAVA Isotonic Regression (In-Sample for Test 1)
            split_t1 = test_datasets['test1']
            X_t1 = split_t1['X_3d'][:, layer_idx, :]
            X_t1_pca = pca.transform(scaler.transform(X_t1))
            y1_t1, y2_t1 = split_t1['y1'], split_t1['y2']

            raw_s_t1 = model.predict_proba(X_t1_pca, y1_t1)[:, 1]

            # 擬合 PAVA 演算法 (Isotonic Regression) — Fit on Test 1
            # 1. 全域 PAVA
            iso_overall = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
            iso_overall.fit(raw_s_t1, y2_t1)

            # 2. y1=0 Subgroup PAVA
            mask0_t1 = (y1_t1 == 0)
            iso_h0 = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
            if np.sum(mask0_t1) > 0 and len(np.unique(y2_t1[mask0_t1])) > 1:
                iso_h0.fit(raw_s_t1[mask0_t1], y2_t1[mask0_t1])
            else:
                iso_h0.fit(raw_s_t1, y2_t1)

            # 3. y1=1 Subgroup PAVA
            mask1_t1 = (y1_t1 == 1)
            iso_h1 = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
            if np.sum(mask1_t1) > 0 and len(np.unique(y2_t1[mask1_t1])) > 1:
                iso_h1.fit(raw_s_t1[mask1_t1], y2_t1[mask1_t1])
            else:
                iso_h1.fit(raw_s_t1, y2_t1)

            calibration_data['layers'][layer][model_name] = {}

            # 對各獨立測試集計算與評估
            for split_name, split_info in test_datasets.items():
                X_sp_3d = split_info['X_3d']
                y1_sp = split_info['y1']
                y2_sp = split_info['y2']

                X_sp = X_sp_3d[:, layer_idx, :]
                X_sp_scaled = scaler.transform(X_sp)
                X_sp_pca = pca.transform(X_sp_scaled)

                # Raw score
                s_raw = model.predict_proba(X_sp_pca, y1_sp)[:, 1]

                # Calibrated probabilities (PAVA)
                prob_cal_overall = iso_overall.transform(s_raw)

                prob_cal_split = np.zeros_like(s_raw)
                mask0_sp = (y1_sp == 0)
                mask1_sp = (y1_sp == 1)
                if np.sum(mask0_sp) > 0:
                    prob_cal_split[mask0_sp] = iso_h0.transform(s_raw[mask0_sp])
                if np.sum(mask1_sp) > 0:
                    prob_cal_split[mask1_sp] = iso_h1.transform(s_raw[mask1_sp])

                # 指標計算
                m_raw = calculate_all_metrics(y2_sp, s_raw)
                m_cal_ov = calculate_all_metrics(y2_sp, prob_cal_overall)
                m_cal_sp = calculate_all_metrics(y2_sp, prob_cal_split)

                brier_impr_ov = (m_raw['brier'] - m_cal_ov['brier']) / m_raw['brier'] if m_raw['brier'] > 0 else 0.0
                brier_impr_sp = (m_raw['brier'] - m_cal_sp['brier']) / m_raw['brier'] if m_raw['brier'] > 0 else 0.0

                calibration_records.append({
                    'layer': layer,
                    'model': model_name,
                    'split': split_name,
                    'brier_raw': m_raw['brier'],
                    'brier_cal_overall': m_cal_ov['brier'],
                    'brier_cal_split': m_cal_sp['brier'],
                    'brier_impr_pct_overall': brier_impr_ov * 100.0,
                    'brier_impr_pct_split': brier_impr_sp * 100.0,
                    'reliability_raw': m_raw['reliability'],
                    'reliability_cal': m_cal_sp['reliability'],
                    'resolution_raw': m_raw['resolution'],
                    'resolution_cal': m_cal_sp['resolution'],
                    'uncertainty': m_raw['uncertainty'],
                    'logloss_raw': m_raw['logloss'],
                    'logloss_cal': m_cal_sp['logloss']
                })

                calibration_data['layers'][layer][model_name][split_name] = {
                    'y1': y1_sp,
                    'y2': y2_sp,
                    's_raw': s_raw,
                    'prob_cal_overall': prob_cal_overall,
                    'prob_cal_split': prob_cal_split,
                    'iso_overall': iso_overall,
                    'iso_h0': iso_h0,
                    'iso_h1': iso_h1,
                    'metrics_raw': m_raw,
                    'metrics_cal_overall': m_cal_ov,
                    'metrics_cal_split': m_cal_sp
                }

                print(f"  │    └─ [{split_name:6s}] Brier Raw: {m_raw['brier']:.4f} ➔ PAVA Cal: {m_cal_sp['brier']:.4f} (改善: {brier_impr_sp:.2%}) | Rel: {m_cal_sp['reliability']:.4f}")

    # 儲存 CSV 與 joblib 數據
    df_res = pd.DataFrame(calibration_records)
    csv_out = os.path.join(output_dir, "framework_calibration_summary.csv")
    df_res.to_csv(csv_out, index=False, encoding='utf-8-sig')

    joblib_out = os.path.join(output_dir, "calibration_data.joblib")
    joblib.dump(calibration_data, joblib_out)

    print("\n" + "=" * 80)
    print(f"階段二 PAVA 機率校準計算完成！數據已儲存至:")
    print(f"  ├─ 數據摘要 CSV: {csv_out}")
    print(f"  └─ 校準完整資料: {joblib_out}")
    print("=" * 80)

if __name__ == "__main__":
    main()
