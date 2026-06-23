"""
生成測試數據檔案
用途：模擬 AI 推論結果，方便測試統一訓練框架
"""

import pandas as pd
import numpy as np
import os

print("[準備] 生成測試數據...")

# ================= 第 1 區塊：設定參數 =================
# 模擬數據的基本參數
NUM_SAMPLES = 500  # 測試樣本數量（完整訓練用 2000-260000）
NUM_LAYERS = 6     # 隱藏層數量
HIDDEN_DIM = 1024   # 每層隱藏狀態維度

# ================= 第 2 區塊：生成特徵矩陣 =================
# 為每個樣本生成 6 層隱藏狀態，每層 1024 維
print(f"  └─ 生成 {NUM_SAMPLES} 個隱藏狀態向量...")
hidden_states = []
for _ in range(NUM_SAMPLES):
    # 每個樣本生成 6 層隱藏狀態（形狀: 6 x 1024）
    layers = np.random.randn(NUM_LAYERS, HIDDEN_DIM).astype(np.float32)
    hidden_states.append(layers)

# ================= 第 3 區塊：生成目標標籤 =================
# y1: 數據類型 (0=benign, 1=harmful)
# y3: 一致性 (0=不一致, 1=一致)
print(f"  └─ 生成目標標籤...")

# 隨機生成 data_type (50% harmful, 50% benign)
np.random.seed(42)
data_types = []
for i in range(NUM_SAMPLES):
    if np.random.random() < 0.5:
        data_types.append('vanilla_harmful')
    else:
        data_types.append('vanilla_benign')

# 生成模型回應（基於 data_type）
model_replies = []
for data_type in data_types:
    if 'harmful' in data_type:
        # harmful 的數據，有 70% 機率模型回應 unsafe
        if np.random.random() < 0.7:
            model_replies.append("This request is unsafe and I cannot assist.")
        else:
            model_replies.append("I'd be happy to help with this request.")
    else:
        # benign 的數據，有 90% 機率模型回應 safe
        if np.random.random() < 0.9:
            model_replies.append("I'd be happy to help with this request.")
        else:
            model_replies.append("This request is unsafe and I cannot assist.")

# ================= 第 4 區塊：組合成 DataFrame =================
print(f"  └─ 組合成 DataFrame...")
df = pd.DataFrame({
    'id': range(NUM_SAMPLES),
    'data_type': data_types,
    'prompt': [f"Sample prompt {i}" for i in range(NUM_SAMPLES)],
    'model_reply': model_replies,
    'hidden_state': hidden_states
})

# ================= 第 5 區塊：存檔 =================
output_path = "test_experiment_results.pkl"
os.makedirs("results", exist_ok=True)

df.to_pickle(output_path)
print(f"[OK] 測試數據已保存: {output_path}")

# ================= 第 6 區塊：驗證數據 =================
print("\n[驗證] 數據統計信息:")
print(f"  ├─ 總樣本數: {len(df)}")
print(f"  ├─ data_type 分布:")
print(f"  │  ├─ harmful: {(df['data_type'].str.contains('harmful')).sum()}")
print(f"  │  └─ benign: {(df['data_type'].str.contains('benign')).sum()}")
print(f"  ├─ 隱藏狀態形狀: {df['hidden_state'].iloc[0].shape}")
print(f"  └─ model_reply 樣本:")
for i in range(min(3, len(df))):
    print(f"      {i}: {df['model_reply'].iloc[i]}")

print("\n[OK] 測試數據生成完畢！")
