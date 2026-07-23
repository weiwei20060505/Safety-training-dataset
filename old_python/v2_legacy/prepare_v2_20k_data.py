import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def prepare_data():
    print("="*60)
    print("v2_20k 資料抽樣與切分作業")
    print("="*60)
    
    raw_path = "data/experiment_results_train.pkl"
    if not os.path.exists(raw_path):
        print(f"錯誤: 找不到原始資料檔案 {raw_path}")
        return
        
    print("[1] 讀取原始母體資料 (85,000 筆)...")
    df_raw = pd.read_pickle(raw_path)
    print(f"  └─ 成功讀取 {len(df_raw)} 筆資料")
    
    print("[2] 計算標籤 y1, y2, y3...")
    y1 = df_raw['model_reply'].str.lower().str.contains('unsafe').astype(int)
    y2 = df_raw['data_type'].str.contains('harmful').astype(int)
    y3 = (y1 == y2).astype(int)
    
    df_raw['y1'] = y1
    df_raw['y2'] = y2
    df_raw['y3'] = y3
    
    print(f"  └─ 原始母體 Y3 正確率比例: {y3.mean():.4f} ({y3.sum()} / {len(y3)})")
    
    print("[3] 依據 Y3 進行分層隨機抽樣 (Stratified Sampling) 抽出 20,000 筆...")
    # Using train_test_split to sample 20,000 rows proportionally
    df_20000, _ = train_test_split(
        df_raw, train_size=20000, random_state=42, stratify=df_raw['y3']
    )
    df_20000 = df_20000.reset_index(drop=True)
    print(f"  └─ 成功抽出 {len(df_20000)} 筆資料，Y3 比例: {df_20000['y3'].mean():.4f}")
    
    print("[4] 切分資料集 (60% Train, 20% Val, 20% Test)...")
    # 60% Train (12,000), 40% Train_Val (8,000)
    df_train, df_val_test = train_test_split(
        df_20000, test_size=0.4, random_state=42, stratify=df_20000['y3']
    )
    # Split 40% into Val (4,000) and Test (4,000)
    df_val, df_test = train_test_split(
        df_val_test, test_size=0.5, random_state=42, stratify=df_val_test['y3']
    )
    
    print("[5] 切分 Test 為 Test1 (校正集 2,000) 與 Test2 (最終評估集 2,000)...")
    df_test1, df_test2 = train_test_split(
        df_test, test_size=0.5, random_state=42, stratify=df_test['y3']
    )
    
    output_dir = "data/v2_20k"
    os.makedirs(output_dir, exist_ok=True)
    
    print("[6] 儲存實體檔案至 data/v2_20k/...")
    df_20000.to_pickle(os.path.join(output_dir, "experiment_results_train_20000.pkl"))
    df_train.to_pickle(os.path.join(output_dir, "train_12000.pkl"))
    df_val.to_pickle(os.path.join(output_dir, "val_4000.pkl"))
    df_test1.to_pickle(os.path.join(output_dir, "test1_2000.pkl"))
    df_test2.to_pickle(os.path.join(output_dir, "test2_2000.pkl"))
    
    print("="*60)
    print("資料抽樣與切分完成！各集合資料筆數：")
    print(f"  - 抽樣全集 (20,000): {len(df_20000)} 筆")
    print(f"  - Train  (12,000): {len(df_train)} 筆")
    print(f"  - Val     (4,000): {len(df_val)} 筆")
    print(f"  - Test1   (2,000): {len(df_test1)} 筆 (校正集)")
    print(f"  - Test2   (2,000): {len(df_test2)} 筆 (評估集)")
    print("="*60)

if __name__ == "__main__":
    prepare_data()
