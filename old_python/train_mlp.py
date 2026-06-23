import os
import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

class DualLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

MODEL_DIR = "results/mlp/models"
IMG_DIR = "results/mlp/images"
LOG_DIR = "results/mlp"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)
sys.stdout = DualLogger(os.path.join(LOG_DIR, "mlp_train_log.txt"))

# 1. 讀取實驗資料
df = pd.read_pickle("experiment_results2.pkl")

# 2. 處理特徵矩陣 X
X_3d = np.array(df['hidden_state'].tolist())
num_layers = X_3d.shape[1]

for i in range(num_layers):
    print(f"\n==================== MLP 處理第 {i+1} 個特徵維度 ====================")
    X_2d = X_3d[:, i, :]  

    # 3. 處理目標變數 y
    y1 = df['data_type'].str.contains('harmful').astype(int) 
    y2 = df['model_reply'].str.lower().str.contains('unsafe').astype(int) 
    y3 = (y1 == y2).astype(int)  

    # 4. 三向切分資料集
    X1_train_val, X1_test, y1_train_val, y1_test = train_test_split(X_2d, y1, test_size=0.2, random_state=42, stratify=y1)
    X3_train_val, X3_test, y3_train_val, y3_test = train_test_split(X_2d, y3, test_size=0.2, random_state=42, stratify=y3)

    X1_train, X1_val, y1_train, y1_val = train_test_split(X1_train_val, y1_train_val, test_size=0.25, random_state=42, stratify=y1_train_val)
    X3_train, X3_val, y3_train, y3_val = train_test_split(X3_train_val, y3_train_val, test_size=0.25, random_state=42, stratify=y3_train_val)

    # 5. 建立 MLP 模型 (MLPClassifier 不直接支援 class_weight，透過 zero_division=0 處理評估警告)
    print("正在訓練 MLP 模型...")
    clf1 = MLPClassifier(hidden_layer_sizes=(128,), max_iter=200, random_state=42, early_stopping=True, validation_fraction=0.2)
    clf3 = MLPClassifier(hidden_layer_sizes=(128,), max_iter=200, random_state=42, early_stopping=True, validation_fraction=0.2)
    
    clf1.fit(X1_train, y1_train)
    clf3.fit(X3_train, y3_train)

    # 6. 輸出訓練與驗證結果
    print(f"\n--- 📈 y1 MLP 模型訓練過程報告 ---")
    print(f"收斂疊代次數: {clf1.n_iter_}")
    print(f"最終訓練集損失: {clf1.loss_:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(clf1.loss_curve_) + 1), clf1.loss_curve_, label='Training Loss (y1)')
    plt.title(f'Layer {i+1} - y1 MLP Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(IMG_DIR, f'mlp_loss_curve_y1_layer_{i+1}.png'))
    plt.close()

    print(f"\n--- 📈 y3 MLP 模型訓練過程報告 ---")
    print(f"收斂疊代次數: {clf3.n_iter_}")
    print(f"最終訓練集損失: {clf3.loss_:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(clf3.loss_curve_) + 1), clf3.loss_curve_, label='Training Loss (y3)')
    plt.title(f'Layer {i+1} - y3 MLP Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(IMG_DIR, f'mlp_loss_curve_y3_layer_{i+1}.png'))
    plt.close()

    # 7. 在測試集上評估表現
    print("\n--- 🎯 y1 MLP 在測試集上的表現 ---")
    y1_pred = clf1.predict(X1_test)
    print(f"y1 的整體準確率 (Accuracy): {accuracy_score(y1_test, y1_pred):.2f}")
    print(classification_report(y1_test, y1_pred, target_names=['Safe (0)', 'Unsafe (1)'], zero_division=0))

    print("\n--- 🎯 y3 MLP 在測試集上的表現 ---")
    y3_pred = clf3.predict(X3_test)
    print(f"y3 的整體準確率 (Accuracy): {accuracy_score(y3_test, y3_pred):.2f}")
    print(classification_report(y3_test, y3_pred, target_names=['Safe (0)', 'Unsafe (1)'], zero_division=0))

    # 8. 儲存模型
    joblib.dump(clf1, os.path.join(MODEL_DIR, f"mlp_probe_model_y1_{i+1}.pkl"))
    joblib.dump(clf3, os.path.join(MODEL_DIR, f"mlp_probe_model_y3_{i+1}.pkl"))
    print(f"\nMLP 模型已儲存至各自資料夾中。")
    print("\n==================== 分隔線 ====================\n")
