import os
import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# 建立雙向日誌記錄器，同時輸出到控制台與檔案
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

# 設定路徑並初始化日誌
MODEL_DIR = "results/lightgbm/models"
IMG_DIR = "results/lightgbm/images"
LOG_DIR = "results/lightgbm"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)
sys.stdout = DualLogger(os.path.join(LOG_DIR, "lgb_train_log.txt"))

# 1. 讀取實驗資料
df = pd.read_pickle("experiment_results2.pkl")

# 2. 處理特徵矩陣 X
X_3d = np.array(df['hidden_state'].tolist())
num_layers = X_3d.shape[1]

for i in range(num_layers):
    print(f"\n==================== LightGBM 處理第 {i+1} 個特徵維度 ====================")
    X_2d = X_3d[:, i, :]  

    # 3. 處理目標變數 y
    y1 = df['data_type'].str.contains('harmful').astype(int) 
    y2 = df['model_reply'].str.lower().str.contains('unsafe').astype(int) 
    y3 = (y1 == y2).astype(int)  

    # 4. 三向切分資料集 (60% 訓練, 20% 驗證, 20% 測試) 
    X1_train_val, X1_test, y1_train_val, y1_test = train_test_split(X_2d, y1, test_size=0.2, random_state=42, stratify=y1)
    X3_train_val, X3_test, y3_train_val, y3_test = train_test_split(X_2d, y3, test_size=0.2, random_state=42, stratify=y3)

    X1_train, X1_val, y1_train, y1_val = train_test_split(X1_train_val, y1_train_val, test_size=0.25, random_state=42, stratify=y1_train_val)
    X3_train, X3_val, y3_train, y3_val = train_test_split(X3_train_val, y3_train_val, test_size=0.25, random_state=42, stratify=y3_train_val)

    # 5. 建立 LightGBM 分類器並配置 Early Stopping
    print("正在訓練 LightGBM 模型...")
    # 解決 y3 類別不平衡問題：加入 class_weight='balanced'
    clf1 = lgb.LGBMClassifier(n_estimators=300, random_state=42, learning_rate=0.05, verbosity=-1, class_weight='balanced')
    clf3 = lgb.LGBMClassifier(n_estimators=300, random_state=42, learning_rate=0.05, verbosity=-1, class_weight='balanced')
    
    # 訓練 y1
    evals_result_y1 = {}
    clf1.fit(
        X1_train, y1_train,
        eval_set=[(X1_val, y1_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=20, verbose=False),
            lgb.record_evaluation(evals_result_y1)
        ]
    )

    # 訓練 y3
    evals_result_y3 = {}
    clf3.fit(
        X3_train, y3_train,
        eval_set=[(X3_val, y3_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=20, verbose=False),
            lgb.record_evaluation(evals_result_y3)
        ]
    )

    # 6. 輸出最佳特徵並繪製學習曲線 (BCE Loss)
    print(f"\n--- 📈 y1 LightGBM 最佳疊代次數: {clf1.best_iteration_} ---")
    val_loss_y1 = evals_result_y1['valid_0']['binary_logloss']

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(val_loss_y1) + 1), val_loss_y1, label='Validation Binary LogLoss (y1)')
    plt.axvline(x=clf1.best_iteration_, color='red', linestyle='--', label=f'Best Iteration: {clf1.best_iteration_}')
    plt.title(f'Layer {i+1} - y1 LightGBM Learning Curve')
    plt.xlabel('Trees (Iterations)')
    plt.ylabel('LogLoss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(IMG_DIR, f'lgb_learning_curve_y1_layer_{i+1}.png'))
    plt.close()

    # 7. 在測試集上評估表現
    print("\n--- 🎯 y1 LightGBM 在測試集上的表現 ---")
    y1_pred = clf1.predict(X1_test)
    print(f"y1 的整體準確率 (Accuracy): {accuracy_score(y1_test, y1_pred):.2f}")
    print(classification_report(y1_test, y1_pred, target_names=['Safe (0)', 'Unsafe (1)'], zero_division=0))

    print("\n--- 🎯 y3 LightGBM 在測試集上的表現 ---")
    y3_pred = clf3.predict(X3_test)
    print(f"y3 的整體準確率 (Accuracy): {accuracy_score(y3_test, y3_pred):.2f}")
    print(classification_report(y3_test, y3_pred, target_names=['Safe (0)', 'Unsafe (1)'], zero_division=0))

    # 8. 儲存模型
    joblib.dump(clf1, os.path.join(MODEL_DIR, f"lgb_probe_model_y1_{i+1}.pkl"))
    joblib.dump(clf3, os.path.join(MODEL_DIR, f"lgb_probe_model_y3_{i+1}.pkl"))
    print(f"\nLightGBM 模型已儲存至各自資料夾中。")
    print("\n==================== 分隔線 ====================\n")
