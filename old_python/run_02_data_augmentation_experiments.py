import os
import sys
import numpy as np
import joblib
import pandas as pd
import gc
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV

# Import from project modules
from unified_train import DataPreprocessor, DataSplitter
import utils_calibration
from wrapper_models import CorrectnessClassifierWrapper

class DualLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")
    def write(self, message):
        try:
            self.terminal.write(message)
        except UnicodeEncodeError:
            encoding = getattr(self.terminal, 'encoding', 'utf-8') or 'utf-8'
            self.terminal.write(message.encode(encoding, errors='replace').decode(encoding))
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def get_calibrated_classifier(clf, method='isotonic'):
    try:
        from sklearn.frozen import FrozenEstimator
        return CalibratedClassifierCV(FrozenEstimator(clf), method=method, cv=None)
    except ImportError:
        return CalibratedClassifierCV(clf, method=method, cv='prefit')

def supplement_dataframe_keep_ratio(df_test, df_pool, target_col, target_size, random_state=42):
    """
    Supplements df_test with non-overlapping rows from df_pool,
    keeping the same positive class ratio, up to target_size.
    """
    y_test = df_test[target_col].values
    n_pos_curr = np.sum(y_test == 1)
    n_neg_curr = np.sum(y_test == 0)
    total_curr = len(df_test)
    
    p = n_pos_curr / total_curr if total_curr > 0 else 0
    
    target_pos = int(round(target_size * p))
    target_neg = target_size - target_pos
    
    delta_pos = target_pos - n_pos_curr
    delta_neg = target_neg - n_neg_curr
    
    if delta_pos < 0:
        delta_pos = 0
        target_neg_adj = int(round(n_pos_curr * (1.0 - p) / p)) if p > 0 else target_size
        delta_neg = max(0, target_neg_adj - n_neg_curr)
    elif delta_neg < 0:
        delta_neg = 0
        target_pos_adj = int(round(n_neg_curr * p / (1.0 - p))) if p < 1 else target_size
        delta_pos = max(0, target_pos_adj - n_pos_curr)
        
    pool_pos = df_pool[df_pool[target_col] == 1]
    pool_neg = df_pool[df_pool[target_col] == 0]
    
    rng = np.random.default_rng(random_state)
    
    selected_pos_idx = rng.choice(pool_pos.index, size=min(delta_pos, len(pool_pos)), replace=False)
    selected_neg_idx = rng.choice(pool_neg.index, size=min(delta_neg, len(pool_neg)), replace=False)
    
    df_supplemented = pd.concat([df_test, df_pool.loc[selected_pos_idx], df_pool.loc[selected_neg_idx]], axis=0)
    df_supplemented = df_supplemented.sample(frac=1.0, random_state=random_state).copy()
    return df_supplemented

def supplement_dataframe_to_ratio(df_test, df_pool, target_col, target_ratio_positive, target_size, random_state=42):
    """
    Supplements df_test with non-overlapping rows from df_pool
    so that the final positive class ratio matches target_ratio_positive
    and total size is target_size (adjusted mathematically if needed).
    """
    y_test = df_test[target_col].values
    n_pos_curr = np.sum(y_test == 1)
    n_neg_curr = np.sum(y_test == 0)
    
    target_pos = int(round(target_size * target_ratio_positive))
    target_neg = target_size - target_pos
    
    delta_pos = target_pos - n_pos_curr
    delta_neg = target_neg - n_neg_curr
    
    if delta_pos < 0:
        delta_pos = 0
        target_neg_adj = int(round(n_pos_curr * (1.0 - target_ratio_positive) / target_ratio_positive)) if target_ratio_positive > 0 else target_size
        delta_neg = max(0, target_neg_adj - n_neg_curr)
    elif delta_neg < 0:
        delta_neg = 0
        target_pos_adj = int(round(n_neg_curr * target_ratio_positive / (1.0 - target_ratio_positive))) if target_ratio_positive < 1 else target_size
        delta_pos = max(0, target_pos_adj - n_pos_curr)
        
    pool_pos = df_pool[df_pool[target_col] == 1]
    pool_neg = df_pool[df_pool[target_col] == 0]
    
    rng = np.random.default_rng(random_state)
    
    selected_pos_idx = rng.choice(pool_pos.index, size=min(delta_pos, len(pool_pos)), replace=False)
    selected_neg_idx = rng.choice(pool_neg.index, size=min(delta_neg, len(pool_neg)), replace=False)
    
    df_supplemented = pd.concat([df_test, df_pool.loc[selected_pos_idx], df_pool.loc[selected_neg_idx]], axis=0)
    df_supplemented = df_supplemented.sample(frac=1.0, random_state=random_state).copy()
    return df_supplemented

def main():
    base_output_dir = "results/correctness_reliability_plots"
    os.makedirs(base_output_dir, exist_ok=True)
    sys.stdout = DualLogger(os.path.join(base_output_dir, "augmented_process_log.txt"))
    
    print("="*80)
    print(" 啟動 Step 2: 資料擴增與分佈對齊實驗管線 (10,000 黃金標準規模)")
    print("="*80)
    
    # 1. Load datasets
    TRAIN_PATH = "experiment_results_train_10000.pkl"
    FULL_PATH = "experiment_results_train.pkl"
    EVAL_PATH = "experiment_results_eval.pkl"
    
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(FULL_PATH) or not os.path.exists(EVAL_PATH):
        print(f"錯誤: 確保 {TRAIN_PATH}、{FULL_PATH} 與 {EVAL_PATH} 存在於當前工作目錄。")
        sys.exit(1)
        
    print(f"[1] 載入基準 10000 訓練集: {TRAIN_PATH}")
    prep_train = DataPreprocessor(TRAIN_PATH)
    df_10000 = prep_train.load_data()
    X_3d_train = prep_train.extract_features()
    y_targets_train = prep_train.create_targets()
    # Pre-calculate target labels for df_10000 using preprocessor results to avoid consistency issues
    df_10000['y1'] = y_targets_train[0]
    df_10000['y2'] = y_targets_train[1]
    df_10000['y3'] = y_targets_train[2]
    
    ALIGNED_TEST1_PATH = "aligned_test1.pkl"
    ALIGNED_TEST2_PATH = "aligned_test2.pkl"
    AUGMENTED_TEST1_PATH = "augmented_test1.pkl"
    AUGMENTED_TEST2_PATH = "augmented_test2.pkl"
    
    use_existing_data = (
        os.path.exists(ALIGNED_TEST1_PATH) and
        os.path.exists(ALIGNED_TEST2_PATH) and
        os.path.exists(AUGMENTED_TEST1_PATH) and
        os.path.exists(AUGMENTED_TEST2_PATH)
    )
    
    if use_existing_data:
        print("[2] 偵測到已存在的擴增與對齊資料集，直接載入使用，跳過載入 85000 全量集...")
        augmented_test1_dict = joblib.load(AUGMENTED_TEST1_PATH)
        augmented_test2_dict = joblib.load(AUGMENTED_TEST2_PATH)
        aligned_test1_dict = joblib.load(ALIGNED_TEST1_PATH)
        aligned_test2_dict = joblib.load(ALIGNED_TEST2_PATH)
    else:
        print(f"[2] 載入 85000 全量訓練集並篩選出未使用的資源池...")
        prep_full = DataPreprocessor(FULL_PATH)
        df_full = prep_full.load_data()
        y1_full, y2_full, y3_full = prep_full.create_targets()
        df_full['y1'] = y1_full
        df_full['y2'] = y2_full
        df_full['y3'] = y3_full
        df_unused = df_full[~df_full['id'].isin(df_10000['id'])].copy()
        del df_full
        gc.collect()
        
        print(f"  └─ 資源池可用數據量: {len(df_unused)} 筆")
    
    print(f"[3] 載入外部驗證集: {EVAL_PATH}")
    prep_eval = DataPreprocessor(EVAL_PATH)
    prep_eval.load_data()
    X_3d_eval = prep_eval.extract_features()
    y_targets_eval = prep_eval.create_targets()
    
    num_layers = X_3d_train.shape[1]
    models = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
    targets = ['y1', 'y2', 'y3']
    target_names = {
        'y1': 'y1 (模型回覆安全性預測)',
        'y2': 'y2 (提示詞有害性預測)',
        'y3': 'y3 (安全判定一致性預測)'
    }
    
    if not use_existing_data:
        augmented_test1_dict = {}
        augmented_test2_dict = {}
        aligned_test1_dict = {}
        aligned_test2_dict = {}
    
    # Target loops
    for target_idx, target_name in enumerate(targets):
        y_train = y_targets_train[target_idx]
        y_eval = y_targets_eval[target_idx]
        
        target_ratio_pos = np.mean(y_eval)
        print("\n" + "="*70)
        print(f"開始處理目標任務: {target_names[target_name]}")
        print(f"  └─ Eval 集中正標記(1)比例 = {target_ratio_pos:.4f}")
        print("="*70)
        
        # Split original df_10000 indices to get identical splits across layers
        train_val_idx, test_idx = train_test_split(
            df_10000.index, test_size=0.2, random_state=42, stratify=y_train
        )
        test1_idx, test2_idx = train_test_split(
            test_idx, test_size=0.5, random_state=42
        )
        
        df_test1 = df_10000.loc[test1_idx].copy()
        df_test2 = df_10000.loc[test2_idx].copy()
        
        if use_existing_data:
            df_test1_aug = augmented_test1_dict[target_name]
            df_test2_aug = augmented_test2_dict[target_name]
            df_test1_align = aligned_test1_dict[target_name]
            df_test2_align = aligned_test2_dict[target_name]
        else:
            # 1. Experiment A (Augmentation - scale to exactly 10,000 keeping original ratio)
            df_test1_aug = supplement_dataframe_keep_ratio(df_test1, df_unused, target_name, 10000, random_state=42)
            df_test2_aug = supplement_dataframe_keep_ratio(df_test2, df_unused, target_name, 10000, random_state=43)
            
            # 2. Experiment B (Alignment - scale to exactly 10,000 matching Eval ratio)
            df_test1_align = supplement_dataframe_to_ratio(df_test1, df_unused, target_name, target_ratio_pos, 10000, random_state=44)
            df_test2_align = supplement_dataframe_to_ratio(df_test2, df_unused, target_name, target_ratio_pos, 10000, random_state=45)
            
            # Save to output dictionaries
            augmented_test1_dict[target_name] = df_test1_aug
            augmented_test2_dict[target_name] = df_test2_aug
            aligned_test1_dict[target_name] = df_test1_align
            aligned_test2_dict[target_name] = df_test2_align
        
        print(f"  [10,000 數據集擴增與對齊完成]:")
        print(f"    ├─ 原始 Test1 筆數: {len(df_test1)} (正例比: {np.mean(df_test1[target_name]):.4f})")
        print(f"    ├─ 擴增 Test1 筆數: {len(df_test1_aug)} (正例比: {np.mean(df_test1_aug[target_name]):.4f})")
        print(f"    └─ 對齊 Test1 筆數: {len(df_test1_align)} (正例比: {np.mean(df_test1_align[target_name]):.4f})")
        
        # Convert to 3D arrays once outside the layer loop (optimization)
        print("  正在預先轉換隱藏狀態為 3D 矩陣...")
        X_3d_test1_orig = np.array(df_test1['hidden_state'].tolist())
        X_3d_test2_orig = np.array(df_test2['hidden_state'].tolist())
        X_3d_test1_aug = np.array(df_test1_aug['hidden_state'].tolist())
        X_3d_test2_aug = np.array(df_test2_aug['hidden_state'].tolist())
        X_3d_test1_align = np.array(df_test1_align['hidden_state'].tolist())
        X_3d_test2_align = np.array(df_test2_align['hidden_state'].tolist())

        # Loop layers
        for layer_idx in range(num_layers):
            layer_num = layer_idx + 1
            print(f"\n--- [第 {layer_num} / {num_layers} 層隱藏狀態] ---")
            
            target_plot_dir = os.path.join(base_output_dir, f"layer_{layer_num}", target_name)
            
            # Extract features for original and evaluated sets
            X_2d_eval = X_3d_eval[:, layer_idx, :]
            y_eval_np = np.array(y_eval)
            
            # Original datasets (for backward compatibility evaluation)
            X_test1_orig = X_3d_test1_orig[:, layer_idx, :]
            y_test1_orig = df_test1[target_name].values
            X_test2_orig = X_3d_test2_orig[:, layer_idx, :]
            y_test2_orig = df_test2[target_name].values
            
            # Exp A features (augmented)
            X_test1_aug = X_3d_test1_aug[:, layer_idx, :]
            y_test1_aug = df_test1_aug[target_name].values
            X_test2_aug = X_3d_test2_aug[:, layer_idx, :]
            y_test2_aug = df_test2_aug[target_name].values
            
            # Exp B features (aligned)
            X_test1_align = X_3d_test1_align[:, layer_idx, :]
            y_test1_align = df_test1_align[target_name].values
            X_test2_align = X_3d_test2_align[:, layer_idx, :]
            y_test2_align = df_test2_align[target_name].values
            
            # Dictionary containers
            cal_data_aug = {}
            cal_data_align = {}
            
            edges_aug_native = {}
            edges_aug_adaptive = {}
            edges_align_native = {}
            edges_align_adaptive = {}
            
            for model_name in models:
                model_path = f"results/unified_training/layer_{layer_num}/{model_name.lower()}_{target_name.lower()}_best.pkl"
                if not os.path.exists(model_path):
                    continue
                    
                clf = joblib.load(model_path)
                
                # Generate correctness targets
                if target_name in ['y1', 'y2']:
                    pred_eval = clf.predict(X_2d_eval)
                    pred_test1_orig = clf.predict(X_test1_orig)
                    pred_test2_orig = clf.predict(X_test2_orig)
                    
                    pred_test1_aug = clf.predict(X_test1_aug)
                    pred_test2_aug = clf.predict(X_test2_aug)
                    
                    pred_test1_align = clf.predict(X_test1_align)
                    pred_test2_align = clf.predict(X_test2_align)
                    
                    y_eval_correct = (pred_eval == y_eval_np).astype(int)
                    y_test1_orig_correct = (pred_test1_orig == y_test1_orig).astype(int)
                    y_test2_orig_correct = (pred_test2_orig == y_test2_orig).astype(int)
                    
                    y_test1_aug_correct = (pred_test1_aug == y_test1_aug).astype(int)
                    y_test2_aug_correct = (pred_test2_aug == y_test2_aug).astype(int)
                    
                    y_test1_align_correct = (pred_test1_align == y_test1_align).astype(int)
                    y_test2_align_correct = (pred_test2_align == y_test2_align).astype(int)
                    
                    clf_wrapped = CorrectnessClassifierWrapper(clf, threshold=0.5)
                else:
                    y_eval_correct = y_eval_np
                    y_test1_orig_correct = y_test1_orig
                    y_test2_orig_correct = y_test2_orig
                    
                    y_test1_aug_correct = y_test1_aug
                    y_test2_aug_correct = y_test2_aug
                    
                    y_test1_align_correct = y_test1_align
                    y_test2_align_correct = y_test2_align
                    
                    clf_wrapped = clf
                
                # ==================== 1. Experiment A (model_aug) ====================
                cal_aug = get_calibrated_classifier(clf_wrapped, method='isotonic')
                cal_aug.fit(X_test1_aug, y_test1_aug_correct)
                
                # Save augmented calibrated model
                joblib.dump(cal_aug, f"results/unified_training/layer_{layer_num}/{model_name.lower()}_{target_name.lower()}_calibrated_augmented.pkl")
                
                prob_aug_test1 = cal_aug.predict_proba(X_test1_aug)[:, 1]
                prob_aug_test2 = cal_aug.predict_proba(X_test2_aug)[:, 1]
                prob_aug_eval = cal_aug.predict_proba(X_2d_eval)[:, 1]
                
                # Backward compatibility predictions
                prob_aug_test1_orig = cal_aug.predict_proba(X_test1_orig)[:, 1]
                prob_aug_test2_orig = cal_aug.predict_proba(X_test2_orig)[:, 1]
                
                cal_data_aug[model_name] = {
                    'data_aug_test1': {'y_true': y_test1_aug_correct, 'y_prob': prob_aug_test1},
                    'data_aug_test2': {'y_true': y_test2_aug_correct, 'y_prob': prob_aug_test2},
                    'data_eval': {'y_true': y_eval_correct, 'y_prob': prob_aug_eval},
                    # Backward compatibility
                    'data_std_test1': {'y_true': y_test1_orig_correct, 'y_prob': prob_aug_test1_orig},
                    'data_std_test2': {'y_true': y_test2_orig_correct, 'y_prob': prob_aug_test2_orig}
                }
                
                edges_aug_native[model_name] = utils_calibration.get_native_bins(prob_aug_test1)
                edges_aug_adaptive[model_name] = utils_calibration.get_adaptive_bins(prob_aug_test1, n_bins=10)
                
                # ==================== 2. Experiment B (model_align) ====================
                cal_align = get_calibrated_classifier(clf_wrapped, method='isotonic')
                cal_align.fit(X_test1_align, y_test1_align_correct)
                
                # Save aligned calibrated model
                joblib.dump(cal_align, f"results/unified_training/layer_{layer_num}/{model_name.lower()}_{target_name.lower()}_calibrated_aligned.pkl")
                
                prob_align_test1 = cal_align.predict_proba(X_test1_align)[:, 1]
                prob_align_test2_new = cal_align.predict_proba(X_test2_align)[:, 1]
                prob_align_eval = cal_align.predict_proba(X_2d_eval)[:, 1]
                
                # Backward compatibility predictions
                prob_align_test1_orig = cal_align.predict_proba(X_test1_orig)[:, 1]
                
                cal_data_align[model_name] = {
                    'data_align_test1': {'y_true': y_test1_align_correct, 'y_prob': prob_align_test1},
                    'data_align_test2': {'y_true': y_test2_align_correct, 'y_prob': prob_align_test2_new},
                    'data_eval': {'y_true': y_eval_correct, 'y_prob': prob_align_eval},
                    # Backward compatibility
                    'data_std_test1': {'y_true': y_test1_orig_correct, 'y_prob': prob_align_test1_orig}
                }
                
                edges_align_native[model_name] = utils_calibration.get_native_bins(prob_align_test1)
                edges_align_adaptive[model_name] = utils_calibration.get_adaptive_bins(prob_align_test1, n_bins=10)
            
            # ==================== Plotting Experiment A (model_aug) ====================
            print("  [繪圖] 繪製 Exp A (model_aug) 內建與動態區間對比圖...")
            aug_dir = os.path.join(target_plot_dir, "model_aug")
            aug_native_dir = os.path.join(aug_dir, "native")
            aug_adaptive_dir = os.path.join(aug_dir, "adaptive")
            
            for ds_key in ['data_aug_test1', 'data_aug_test2', 'data_eval', 'data_std_test1', 'data_std_test2']:
                ds_plot_data = {m: cal_data_aug[m][ds_key] for m in cal_data_aug}
                
                # Native
                utils_calibration.plot_comparison_line(
                    ds_plot_data, edges_aug_native,
                    f"第 {layer_num} 層 - {target_names[target_name]} (model_aug on {ds_key} - 內建區間)",
                    os.path.join(aug_native_dir, f"model_aug_on_{ds_key}_bin_native_lines.png")
                )
                utils_calibration.plot_side_by_side_bars(
                    ds_plot_data, edges_aug_native,
                    f"第 {layer_num} 層 - {target_names[target_name]} (model_aug on {ds_key} - 內建區間)",
                    os.path.join(aug_native_dir, f"model_aug_on_{ds_key}_bin_native_bars.png")
                )
                
                # Adaptive
                utils_calibration.plot_comparison_line(
                    ds_plot_data, edges_aug_adaptive,
                    f"第 {layer_num} 層 - {target_names[target_name]} (model_aug on {ds_key} - 動態區間)",
                    os.path.join(aug_adaptive_dir, f"model_aug_on_{ds_key}_bin_adaptive_lines.png")
                )
                utils_calibration.plot_side_by_side_bars(
                    ds_plot_data, edges_aug_adaptive,
                    f"第 {layer_num} 層 - {target_names[target_name]} (model_aug on {ds_key} - 動態區間)",
                    os.path.join(aug_adaptive_dir, f"model_aug_on_{ds_key}_bin_adaptive_bars.png")
                )
                
            # ==================== Plotting Experiment B (model_align) ====================
            print("  [繪圖] 繪製 Exp B (model_align) 內建與動態區間對比圖...")
            align_dir = os.path.join(target_plot_dir, "model_align")
            align_native_dir = os.path.join(align_dir, "native")
            align_adaptive_dir = os.path.join(align_dir, "adaptive")
            
            for ds_key in ['data_align_test1', 'data_align_test2', 'data_eval', 'data_std_test1']:
                ds_plot_data = {m: cal_data_align[m][ds_key] for m in cal_data_align}
                
                # Native
                utils_calibration.plot_comparison_line(
                    ds_plot_data, edges_align_native,
                    f"第 {layer_num} 層 - {target_names[target_name]} (model_align on {ds_key} - 內建區間)",
                    os.path.join(align_native_dir, f"model_align_on_{ds_key}_bin_native_lines.png")
                )
                utils_calibration.plot_side_by_side_bars(
                    ds_plot_data, edges_align_native,
                    f"第 {layer_num} 層 - {target_names[target_name]} (model_align on {ds_key} - 內建區間)",
                    os.path.join(align_native_dir, f"model_align_on_{ds_key}_bin_native_bars.png")
                )
                
                # Adaptive
                utils_calibration.plot_comparison_line(
                    ds_plot_data, edges_align_adaptive,
                    f"第 {layer_num} 層 - {target_names[target_name]} (model_align on {ds_key} - 動態區間)",
                    os.path.join(align_adaptive_dir, f"model_align_on_{ds_key}_bin_adaptive_lines.png")
                )
                utils_calibration.plot_side_by_side_bars(
                    ds_plot_data, edges_align_adaptive,
                    f"第 {layer_num} 層 - {target_names[target_name]} (model_align on {ds_key} - 動態區間)",
                    os.path.join(align_adaptive_dir, f"model_align_on_{ds_key}_bin_adaptive_bars.png")
                )
                
    # 3. Save physical files for augmented and aligned datasets (Target 1)
    print("\n[存檔] 正在將 10,000 規模資料集儲存至實體 pickle 檔案...")
    joblib.dump(augmented_test1_dict, "augmented_test1.pkl")
    joblib.dump(augmented_test2_dict, "augmented_test2.pkl")
    joblib.dump(aligned_test1_dict, "aligned_test1.pkl")
    joblib.dump(aligned_test2_dict, "aligned_test2.pkl")
    print("  └─ 成功儲存 augmented_test1.pkl, augmented_test2.pkl, aligned_test1.pkl, aligned_test2.pkl")

    print("\n" + "="*80)
    print(" Step 2: 運行完成！10,000 規模擴增/自適應校正實驗與回溯測試圖表已完全生成。")
    print("="*80)

if __name__ == "__main__":
    main()
