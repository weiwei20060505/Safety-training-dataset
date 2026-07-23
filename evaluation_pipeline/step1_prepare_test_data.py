import os
import sys
import numpy as np
import pandas as pd
import joblib
import gc
from sklearn.model_selection import train_test_split

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unified_train import DataPreprocessor

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

def main():
    print("="*80)
    # 這裡只做 10,000 筆的資料擴增，不區分 aug 和 align 任務了
    print(" [步驟 1] 開始測試集分割與擴增 (統一擴增至 10,000 筆，不分 aug/align)")
    print("="*80)
    
    TRAIN_PATH = "data/experiment_results_train_10000.pkl"
    FULL_PATH = "data/experiment_results_train.pkl"
    
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(FULL_PATH):
        print(f"錯誤: 確保 {TRAIN_PATH} 與 {FULL_PATH} 存在。")
        sys.exit(1)
        
    print(f"1. 載入基準 10,000 訓練集...")
    prep_train = DataPreprocessor(TRAIN_PATH)
    df_10000 = prep_train.load_data()
    y_targets_train = prep_train.create_targets()
    df_10000['y1'] = y_targets_train[0]
    df_10000['y2'] = y_targets_train[1]
    df_10000['y3'] = y_targets_train[2]
    
    print(f"2. 載入 85,000 全量訓練集以過濾出可用資源池...")
    prep_full = DataPreprocessor(FULL_PATH)
    df_full = prep_full.load_data()
    y_full = prep_full.create_targets()
    df_full['y1'] = y_full[0]
    df_full['y2'] = y_full[1]
    df_full['y3'] = y_full[2]
    
    df_unused = df_full[~df_full['id'].isin(df_10000['id'])].copy()
    del df_full
    gc.collect()
    print(f"  └─ 資源池可用無重複數據量: {len(df_unused)} 筆")
    
    test1_dict = {}
    test2_dict = {}
    
    targets = ['y1', 'y2', 'y3']
    for target_idx, target_name in enumerate(targets):
        y_train = y_targets_train[target_idx]
        print(f"\n處理目標任務: {target_name}")
        
        # 進行分割 (維持 60:20:20 中的 20% test，即 2,000 筆)
        train_val_idx, test_idx = train_test_split(
            df_10000.index, test_size=0.2, random_state=42, stratify=y_train
        )
        test1_idx, test2_idx = train_test_split(
            test_idx, test_size=0.5, random_state=42
        )
        
        df_test1 = df_10000.loc[test1_idx].copy()
        df_test2 = df_10000.loc[test2_idx].copy()
        
        # 擴增至 10,000 筆，保持原始比例，確保資料不重複
        df_test1_aug = supplement_dataframe_keep_ratio(df_test1, df_unused, target_name, 10000, random_state=42)
        df_test2_aug = supplement_dataframe_keep_ratio(df_test2, df_unused, target_name, 10000, random_state=43)
        
        test1_dict[target_name] = df_test1_aug
        test2_dict[target_name] = df_test2_aug
        
        print(f"  └─ {target_name} 原始 Test1: {len(df_test1)} (正比: {np.mean(y_train.loc[test1_idx]):.4f})")
        print(f"  └─ {target_name} 擴增 Test1: {len(df_test1_aug)} (正比: {np.mean(df_test1_aug[target_name]):.4f})")
        print(f"  └─ {target_name} 原始 Test2: {len(df_test2)} (正比: {np.mean(y_train.loc[test2_idx]):.4f})")
        print(f"  └─ {target_name} 擴增 Test2: {len(df_test2_aug)} (正比: {np.mean(df_test2_aug[target_name]):.4f})")

    # 儲存結果
    joblib.dump(test1_dict, "data/test1.pkl")
    joblib.dump(test2_dict, "data/test2.pkl")
    print("\n[OK] 資料分割與擴增完成！儲存於 data/test1.pkl 與 data/test2.pkl")

if __name__ == '__main__':
    main()
