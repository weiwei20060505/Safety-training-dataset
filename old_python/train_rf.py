import os
import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

MODEL_DIR = "results/random_forest/models"
IMG_DIR = "results/random_forest/images"
LOG_DIR = "results/random_forest"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)
sys.stdout = DualLogger(os.path.join(LOG_DIR, "rf_train_log.txt"))

# 1. 讀取實驗資料
df = pd.read_pickle("experiment_results2.pkl")

# 2. 處理特徵矩陣 X
X_3d = np.array(df['hidden_state'].tolist())
num_layers = X_3d.shape[1]

for i in range(num_layers):
    print(f"\n==================== Random Forest 處理第 {i+1} 個特徵維度 ====================")
    X_2d = X_3d[:, i, :]  

    # 3. 處理目標變數 y
    y1 = df['data_type'].str.contains('harmful').astype(int) 
    y2 = df['model_reply'].str.lower().str.contains('unsafe').astype(int) 
    y3 = (y1 == y2).astype(int)  

    # 4. 切分 Train / Test
    X1_train, X1_test, y1_train, y1_test = train_test_split(X_2d, y1, test_size=0.2, random_state=42, stratify=y1)
    X3_train, X3_test, y3_train, y3_test = train_test_split(X_2d, y3, test_size=0.2, random_state=42, stratify=y3)

    # 5. 建立並訓練隨機森林模型，加入 class_weight='balanced' 修正不平衡
    print("正在訓練隨機森林模型...")
    clf1 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, class_weight='balanced')
    clf3 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, class_weight='balanced')
    
    clf1.fit(X1_train, y1_train)
    clf3.fit(X3_train, y3_train)

    # 6. 繪製前 20 個最重要的特徵維度圖
    print(f"\n--- 📈 繪製特徵重要性圖表 ---")
    importances_y1 = clf1.feature_importances_
    indices_y1 = np.argsort(importances_y1)[::-1][:20]

    plt.figure(figsize=(10, 5))
    plt.title(f'Layer {i+1} - Top 20 Feature Importances (y1)')
    plt.bar(range(20), importances_y1[indices_y1], align="center")
    plt.xticks(range(20), indices_y1, rotation=45)
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, f'rf_feature_importance_y1_layer_{i+1}.png'))
    plt.close()

    # 7. 在測試集上評估表現
    print("\n--- 🎯 y1 Random Forest 在測試集上的表現 ---")
    y1_pred = clf1.predict(X1_test)
    print(f"y1 的整體準確率 (Accuracy): {accuracy_score(y1_test, y1_pred):.2f}")
    print(classification_report(y1_test, y1_pred, target_names=['Safe (0)', 'Unsafe (1)'], zero_division=0))

    print("\n--- 🎯 y3 Random Forest 在測試集上的表現 ---")
    y3_pred = clf3.predict(X3_test)
    print(f"y3 的整體準確率 (Accuracy): {accuracy_score(y3_test, y3_pred):.2f}")
    print(classification_report(y3_test, y3_pred, target_names=['Safe (0)', 'Unsafe (1)'], zero_division=0))

    # 8. 儲存模型
    joblib.dump(clf1, os.path.join(MODEL_DIR, f"rf_probe_model_y1_{i+1}.pkl"))
    joblib.dump(clf3, os.path.join(MODEL_DIR, f"rf_probe_model_y3_{i+1}.pkl"))
    print(f"\n隨機森林模型已儲存至各自資料夾中。")
    print("\n==================== 分隔線 ====================\n")
