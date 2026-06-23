"""
修正方案：解決三個統計陷阱

問題 1：維度災難 - 高維空間完美分離
問題 2：類別不平衡 - 模型無腦預測多數類
問題 3：數據洩漏 - 基礎模型背答案

此檔案生成改進的訓練數據，使用交叉驗證避免數據洩漏
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
import os

print("="*80)
print("修正三大統計陷阱 - 生成高質量訓練數據")
print("="*80)

# ================= 第 1 區塊：生成基礎特徵 =================
print("\n[第 1 步] 生成基礎特徵...")

NUM_SAMPLES = 2000  # 更大的樣本數，減輕維度災難
NUM_LAYERS = 6
HIDDEN_DIM = 1024  # 高維特徵

np.random.seed(42)

# 生成隱藏狀態
print(f"  ├─ 生成 {NUM_SAMPLES} 個樣本")
print(f"  ├─ 維度: {NUM_LAYERS} 層 × {HIDDEN_DIM} 維 = {NUM_LAYERS * HIDDEN_DIM} 特徵")
print(f"  └─ 樣本數/維度比: {NUM_SAMPLES}/{HIDDEN_DIM} = {NUM_SAMPLES/HIDDEN_DIM:.2f}")

hidden_states = []
for _ in range(NUM_SAMPLES):
    layers = np.random.randn(NUM_LAYERS, HIDDEN_DIM).astype(np.float32)
    hidden_states.append(layers)

# ================= 第 2 區塊：生成標籤 =================
print("\n[第 2 步] 生成目標變數...")

# 生成提示詞類型 (y_true)
data_types = np.random.choice(['vanilla_harmful', 'vanilla_benign'], NUM_SAMPLES, p=[0.5, 0.5])
y_true = (np.array(data_types) == 'vanilla_harmful').astype(int)

print(f"  ├─ y_true (提示詞有害性):")
print(f"  │  ├─ harmful: {y_true.sum()}")
print(f"  │  └─ benign: {(1-y_true).sum()}")

# ================= 第 3 區塊：MECE 修正 =================
# 🚨 陷阱 1 修正：L2 正則化 (減輕維度災難)
# 🚨 陷阱 2 修正：class_weight='balanced' (處理類別不平衡)
# 🚨 陷阱 3 修正：交叉驗證 (避免數據洩漏)

print("\n[第 3 步] 使用交叉驗證生成客觀預測（避免數據洩漏）...")

# 準備第一層的特徵矩陣
X = np.array([state[0, :] for state in hidden_states])  # 取第一層特徵

print(f"  ├─ 特徵矩陣形狀: {X.shape}")
print(f"  ├─ 樣本數 (N): {X.shape[0]}")
print(f"  ├─ 特徵數 (d): {X.shape[1]}")
print(f"  └─ N/d 比例: {X.shape[0]/X.shape[1]:.4f} (>1 為安全)")

# ===== 關鍵修正：使用交叉驗證 =====
# 這樣每一筆預測都來自「沒見過該樣本」的模型
print(f"\n  ├─ 執行 5-fold 交叉驗證...")

# 建立基礎模型，加入 L2 正則化 (C=0.01 表示強正則化)
base_clf = LogisticRegression(
    C=0.01,              # 🚨 陷阱 1：低 C 值強制 L2 正則化，防止過擬合
    penalty='l2',
    class_weight='balanced',  # 🚨 陷阱 2：平衡類別權重
    max_iter=1000,
    random_state=42
)

# 使用交叉驗證得到客觀預測
objective_predictions = cross_val_predict(
    base_clf, X, y_true, cv=5  # 🚨 陷阱 3：交叉驗證避免數據洩漏
)

print(f"  └─ 交叉驗證完成")

# ================= 第 4 區塊：計算預測錯誤 =================
print("\n[第 4 步] 計算預測錯誤標籤...")

# y_err：基礎模型是否預測錯誤
y_err = (objective_predictions != y_true).astype(int)

print(f"  ├─ 預測準確率: {(y_err == 0).sum() / len(y_err):.2%}")
print(f"  ├─ 預測錯誤率: {y_err.sum() / len(y_err):.2%}")
print(f"  ├─ 錯誤樣本數: {y_err.sum()}")

# 檢查類別平衡
if y_err.sum() < 10:
    print(f"\n  ⚠️  警告：只有 {y_err.sum()} 筆錯誤樣本")
    print(f"     這可能導致嚴重的類別不平衡問題！")
elif y_err.sum() < 50:
    print(f"\n  ⚠️  注意：錯誤樣本相對較少 ({y_err.sum()} 筆)")
    print(f"     class_weight='balanced' 將會很重要")
else:
    print(f"\n  [OK] 錯誤樣本充足，類別平衡情況良好")

# ================= 第 5 區塊：組合成 DataFrame =================
print("\n[第 5 步] 組合成訓練數據...")

df = pd.DataFrame({
    'id': range(NUM_SAMPLES),
    'data_type': data_types,
    'y_true': y_true,
    'objective_prediction': objective_predictions,
    'is_error': y_err,
    'hidden_state': hidden_states
})

print(f"  ├─ DataFrame 形狀: {df.shape}")
print(f"  ├─ 欄位: {list(df.columns)}")

# ================= 第 6 區塊：統計摘要 =================
print("\n[第 6 步] 統計摘要...")

print(f"\n  y_true (提示詞有害性):")
print(f"    ├─ 0 (benign): {(df['y_true']==0).sum()}")
print(f"    └─ 1 (harmful): {(df['y_true']==1).sum()}")

print(f"\n  is_error (預測是否錯誤):")
print(f"    ├─ 0 (正確): {(df['is_error']==0).sum()}")
print(f"    └─ 1 (錯誤): {(df['is_error']==1).sum()}")

# 交叉表：真實值 vs 預測值
print(f"\n  交叉表（真實 vs 預測）:")
confusion_matrix = pd.crosstab(
    df['y_true'],
    df['objective_prediction'],
    rownames=['y_true'],
    colnames=['prediction']
)
print(f"    {confusion_matrix}")

# ================= 第 7 區塊：存檔 =================
output_path = "improved_experiment_results.pkl"
os.makedirs("results", exist_ok=True)

df.to_pickle(output_path)
print(f"\n[完成] 改進的訓練數據已保存: {output_path}")

# ================= 第 8 區塊：使用說明 =================
print("\n" + "="*80)
print("使用說明")
print("="*80)

print("""
使用說明

在 unified_train.py 中進行以下修改：

1. 修改數據路徑（第 508 行）：
   DATA_PATH = "test_experiment_results.pkl"
   改為：
   DATA_PATH = "improved_experiment_results.pkl"

2. 修改模型超參數已自動應用在新版 unified_train.py 中

3. 三個陷阱已在本檔案中修正：
   [修正 1] N/d = 1.95 (改善至 > 1)
   [修正 2] class_weight='balanced' 已應用
   [修正 3] 使用交叉驗證避免數據洩漏

執行訓練：
   python unified_train.py
""")

print("="*80 + "\n")
