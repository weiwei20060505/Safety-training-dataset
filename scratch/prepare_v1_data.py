import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

def main():
    print("="*80)
    print(" [Step 1] Preparing V1 Golden Split Datasets from 85k Pool")
    print("="*80)
    
    full_path = "data/experiment_results_train.pkl"
    if not os.path.exists(full_path):
        print(f"Error: {full_path} not found!")
        sys.exit(1)
        
    print(f"1. Loading 85,000 full training dataset from {full_path} ...")
    df_full = pd.read_pickle(full_path)
    total_samples = len(df_full)
    print(f"  └─ Total samples loaded: {total_samples}")
    
    # Create y1, y2, y3 target columns if not present
    df_full['y1'] = df_full['model_reply'].str.lower().str.contains('unsafe').astype(int)
    df_full['y2'] = df_full['data_type'].str.contains('harmful').astype(int)
    df_full['y3'] = (df_full['y1'] == df_full['y2']).astype(int)
    
    # Target sizes:
    # Train: 77,999 (approx 78k)
    # Val: 2,000
    # Test1: 2,000
    # Test2: 3,000
    # Total = 84,999
    
    val_test_size = 7000
    train_size = total_samples - val_test_size
    
    print(f"\n2. Splitting into Train ({train_size}) and Val+Test ({val_test_size}) ...")
    df_train_full, df_rem = train_test_split(
        df_full, test_size=val_test_size, random_state=42, stratify=df_full['y2']
    )
    
    print(f"3. Splitting Val+Test into Val (2,000), Test1 (2,000), Test2 (3,000) ...")
    df_val, df_test_rem = train_test_split(
        df_rem, test_size=5000, random_state=42, stratify=df_rem['y2']
    )
    
    df_test1, df_test2 = train_test_split(
        df_test_rem, test_size=3000, random_state=42, stratify=df_test_rem['y2']
    )
    
    print(f"\nSplit Summary:")
    print(f"  ├─ v1_train_full: {len(df_train_full)} samples (Harmful ratio: {np.mean(df_train_full['y2']):.4f})")
    print(f"  ├─ v1_val:        {len(df_val)} samples (Harmful ratio: {np.mean(df_val['y2']):.4f})")
    print(f"  ├─ v1_test1:      {len(df_test1)} samples (Harmful ratio: {np.mean(df_test1['y2']):.4f})")
    print(f"  └─ v1_test2:      {len(df_test2)} samples (Harmful ratio: {np.mean(df_test2['y2']):.4f})")
    
    # Save files
    os.makedirs("data", exist_ok=True)
    
    print("\n4. Saving dataset files to data/ ...")
    df_train_full.to_pickle("data/v1_train_full.pkl")
    df_val.to_pickle("data/v1_val.pkl")
    
    dict_test1 = {'y1': df_test1, 'y2': df_test1, 'y3': df_test1}
    dict_test2 = {'y1': df_test2, 'y2': df_test2, 'y3': df_test2}
    
    joblib.dump(dict_test1, "data/v1_test1.pkl")
    joblib.dump(dict_test2, "data/v1_test2.pkl")
    
    # Also save to data/test1.pkl and data/test2.pkl for standard pipeline compatibility
    joblib.dump(dict_test1, "data/test1.pkl")
    joblib.dump(dict_test2, "data/test2.pkl")
    
    print("[OK] All V1 dataset files generated successfully!")

if __name__ == "__main__":
    main()
