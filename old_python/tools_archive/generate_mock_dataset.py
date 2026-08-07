"""
生成實驗測試數據檔案 (Mock Dataset Generator - 高覆蓋率特徵版)
用於無原始 GPU 推論資料時，測試整套訓練與校正管線 (unified_train.py -> step1 -> step2 -> step3)
"""

import os
import sys
import io
import pandas as pd
import numpy as np

# 處理 Windows 控制台 cp950 Unicode 編碼相容性
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def generate_mock_df(num_samples=1000, num_layers=6, hidden_dim=1024, seed=42):
    np.random.seed(seed)
    print(f"正在生成 {num_samples} 筆測試資料 ({num_layers} 層 x {hidden_dim} 維)...")
    
    # 隨機產生 data_type (Harmful vs Benign)
    data_types = np.random.choice(
        ['vanilla_harmful', 'vanilla_benign', 'adversarial_harmful', 'adversarial_benign'],
        size=num_samples,
        p=[0.3, 0.3, 0.2, 0.2]
    )
    
    is_harmful = np.array(['harmful' in dt for dt in data_types])
    
    # 生成模型回覆 (根據 Prompt 有害性設置動態反應)
    model_replies = []
    for h in is_harmful:
        if h:
            prob_unsafe = np.random.uniform(0.6, 0.95)
        else:
            prob_unsafe = np.random.uniform(0.05, 0.4)
            
        if np.random.random() < prob_unsafe:
            model_replies.append("This request is unsafe and I cannot assist.")
        else:
            model_replies.append("I can help with that request.")
            
    # 建立具有連續分佈特徵的隱藏狀態 (確保探針模型可在 [0, 1] 產生連續平滑的機率分佈)
    hidden_states = []
    for i in range(num_samples):
        # 加入 signal (標籤與隱藏狀態的相關性) + noise
        signal = 0.8 * (1.0 if is_harmful[i] else -1.0)
        layers = []
        for l in range(num_layers):
            layer_vec = np.random.randn(hidden_dim).astype(np.float32) * 1.5
            layer_vec[:100] += signal * (l + 1) / 3.0  # 漸進層次特徵
            layers.append(layer_vec)
        hidden_states.append(np.array(layers))
        
    df = pd.DataFrame({
        'id': range(num_samples),
        'data_type': data_types,
        'prompt': [f"Sample prompt {i}" for i in range(num_samples)],
        'model_reply': model_replies,
        'hidden_state': hidden_states
    })
    
    return df

def main():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. 生成 1,000 筆基準訓練集
    train_10k_path = os.path.join(data_dir, "experiment_results_train_10000.pkl")
    df_10k = generate_mock_df(num_samples=1000, seed=42)
    df_10k.to_pickle(train_10k_path)
    print(f"[OK] 已生成基準訓練集: {train_10k_path}")
        
    # 2. 生成全量資源池 (2,500 筆)
    full_path = os.path.join(data_dir, "experiment_results_train.pkl")
    df_full = generate_mock_df(num_samples=2500, seed=123)
    df_full.to_pickle(full_path)
    print(f"[OK] 已生成全量資源池: {full_path}")

    # 3. 生成 Eval 獨立評估集 (500 筆)
    eval_path = os.path.join(data_dir, "experiment_results_eval.pkl")
    df_eval = generate_mock_df(num_samples=500, seed=999)
    df_eval.to_pickle(eval_path)
    print(f"[OK] 已生成評估資料集: {eval_path}")

if __name__ == "__main__":
    main()
