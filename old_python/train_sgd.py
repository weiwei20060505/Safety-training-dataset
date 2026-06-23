import os
import sys
import copy
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss, classification_report, accuracy_score

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

MODEL_DIR = "results/sgd/models"
IMG_DIR = "results/sgd/images"
LOG_DIR = "results/sgd"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)
sys.stdout = DualLogger(os.path.join(LOG_DIR, "sgd_train_log.txt"))

# 1. 讀取實驗資料
df = pd.read_pickle("experiment_results2.pkl")

# 2. 處理特徵矩陣 X
X_3d = np.array(df['hidden_state'].tolist())
num_layers = X_3d.shape[1]

for i in range(num_layers):
    print(f"\n==================== SGD 處理第 {i+1} 個特徵維度 ====================")
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

    # 5. 建立支援逐筆訓練的羅吉斯迴歸，加入 class_weight='balanced'
    print("正在逐筆訓練羅吉斯迴歸並追蹤 BCE Loss...")
    clf1 = SGDClassifier(loss='log_loss', random_state=42, learning_rate='constant', eta0=0.01, class_weight='balanced')
    clf3 = SGDClassifier(loss='log_loss', random_state=42, learning_rate='constant', eta0=0.01, class_weight='balanced')
    
    val_losses_y1 = []
    best_loss_y1 = float('inf')
    best_step_y1 = 0
    best_model_y1 = None

    val_losses_y3 = []
    best_loss_y3 = float('inf')
    best_step_y3 = 0
    best_model_y3 = None

    classes_y1 = np.array([0, 1])
    classes_y3 = np.array([0, 1])

    y1_train_np = np.array(y1_train)
    y3_train_np = np.array(y3_train)

    # === 新增：設定疊代與批次超參數 ===
    epochs = 50
    batch_size = 64
    n_samples_y1 = len(X1_train)

    print("開始以 Epoch 形式訓練 y1 模型...")
    for epoch in range(epochs):
        # 1. 每個 Epoch 開始前，先打亂訓練資料
        indices = np.random.permutation(n_samples_y1)
        X1_train_shuffled = X1_train[indices]
        y1_train_shuffled = y1_train_np[indices]
        
        # 2. 切分 Mini-batch 進行訓練
        for start_idx in range(0, n_samples_y1, batch_size):
            end_idx = min(start_idx + batch_size, n_samples_y1)
            X_batch = X1_train_shuffled[start_idx:end_idx]
            y_batch = y1_train_shuffled[start_idx:end_idx]
            
            # 模型針對這一個 Batch 的資料進行權重更新
            clf1.partial_fit(X_batch, y_batch, classes=classes_y1)
            
        # 3. 在一整個 Epoch (看過所有資料一輪) 結束後，才計算一次驗證集 Loss
        val_proba = clf1.predict_proba(X1_val)
        current_val_loss = log_loss(y1_val, val_proba, labels=classes_y1)
        val_losses_y1.append(current_val_loss)
        
        # 4. 紀錄最佳模型
        if current_val_loss < best_loss_y1:
            best_loss_y1 = current_val_loss
            best_step_y1 = epoch + 1  # 這裡的 best_step 意義變成了 best_epoch
            best_model_y1 = copy.deepcopy(clf1)

    # === 新增：設定疊代與批次超參數 ===
    epochs = 50
    batch_size = 64
    n_samples_y3 = len(X3_train)

    print("開始以 Epoch 形式訓練 y3 模型...")
    for epoch in range(epochs):
        # 1. 每個 Epoch 開始前，先打亂訓練資料
        indices = np.random.permutation(n_samples_y3)
        X3_train_shuffled = X3_train[indices]
        y3_train_shuffled = y3_train_np[indices]

        # 2. 切分 Mini-batch 進行訓練
        for start_idx in range(0, n_samples_y3, batch_size):
            end_idx = min(start_idx + batch_size, n_samples_y3)
            X_batch = X3_train_shuffled[start_idx:end_idx]
            y_batch = y3_train_shuffled[start_idx:end_idx]

            # 模型針對這一個 Batch 的資料進行權重更新
            clf3.partial_fit(X_batch, y_batch, classes=classes_y3)

        # 3. 在一整個 Epoch (看過所有資料一輪) 結束後，才計算一次驗證集 Loss
        val_proba = clf3.predict_proba(X3_val)
        current_val_loss = log_loss(y3_val, val_proba, labels=classes_y3)
        val_losses_y3.append(current_val_loss)

        # 4. 紀錄最佳模型
        if current_val_loss < best_loss_y3:
            best_loss_y3 = current_val_loss
            best_step_y3 = epoch + 1  # 這裡的 best_step 意義變成了 best_epoch
            best_model_y3 = copy.deepcopy(clf3)
        val_proba = clf3.predict_proba(X3_val)
        current_val_loss = log_loss(y3_val, val_proba, labels=classes_y3)
        val_losses_y3.append(current_val_loss)
        
        if current_val_loss < best_loss_y3:
            best_loss_y3 = current_val_loss
            best_step_y3 = step + 1  
            best_model_y3 = copy.deepcopy(clf3)

    # 6. 輸出結果與畫圖
    print(f"\n--- 📈 y1 模型訓練過程報告 ---")
    print(f"最低 Validation BCE Loss: {best_loss_y1:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(val_losses_y1) + 1), val_losses_y1, label='Validation BCE Loss (y1)')
    plt.axvline(x=best_step_y1, color='red', linestyle='--', label=f'Best Step: {best_step_y1}')
    plt.title(f'Layer {i+1} - y1 Learning Curve')
    plt.xlabel('Training Steps')
    plt.ylabel('BCE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(IMG_DIR, f'learning_curve_y1_layer_{i+1}.png'))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(val_losses_y3) + 1), val_losses_y3, label='Validation BCE Loss (y3)')
    plt.axvline(x=best_step_y3, color='red', linestyle='--', label=f'Best Step: {best_step_y3}')
    plt.title(f'Layer {i+1} - y3 Learning Curve')
    plt.xlabel('Training Steps')
    plt.ylabel('BCE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(IMG_DIR, f'learning_curve_y3_layer_{i+1}.png'))
    plt.close()

    # 7. 評估
    print("\n--- 🎯 y1 最佳模型在測試集上的表現 ---")
    y1_pred = best_model_y1.predict(X1_test)
    print(f"y1 的整體準確率 (Accuracy): {accuracy_score(y1_test, y1_pred):.2f}")
    print(classification_report(y1_test, y1_pred, target_names=['Safe (0)', 'Unsafe (1)'], zero_division=0))

    print("\n--- 🎯 y3 最佳模型在測試集上的表現 ---")
    y3_pred = best_model_y3.predict(X3_test)
    print(f"y3 的整體準確率 (Accuracy): {accuracy_score(y3_test, y3_pred):.2f}")
    print(classification_report(y3_test, y3_pred, target_names=['Safe (0)', 'Unsafe (1)'], zero_division=0))

    # 8. 儲存
    joblib.dump(best_model_y1, os.path.join(MODEL_DIR, f"sgd_probe_model_y1_{i+1}.pkl"))
    joblib.dump(best_model_y3, os.path.join(MODEL_DIR, f"sgd_probe_model_y3_{i+1}.pkl"))
    print(f"\nSGD 最佳模型已儲存至各自資料夾中。")
    print("\n==================== 分隔線 ====================\n")
