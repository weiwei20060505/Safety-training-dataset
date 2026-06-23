import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. 讀取實驗資料
df = pd.read_pickle("experiment_results2.pkl")

# 2. 處理特徵矩陣 X (將 3D 張量降維成 2D 矩陣)
X_3d = np.array(df['hidden_state'].tolist())
num_layers = X_3d.shape[1]
for i in range(num_layers):
    print(f"\n--- 處理第 {i+1} 個特徵維度 ---")
    X_2d = X_3d[:, i, :]  

    # 3. 處理目標變數 y (轉換為 0 和 1)

    #針對 data_type 進行二元分類，harmful 為 1，benign 為 0
    y1 = df['data_type'].str.contains('harmful').astype(int) 
    
    #針對 model_reply 進行二元分類，unsafe 為 1，safe 為 0
    y2 = df['model_reply'].str.lower().str.contains('unsafe').astype(int) 

    # 1 代表一致(Match), 0 代表不一致(Mismatch)
    y3 = (y1 == y2).astype(int)  

    # 4. 三向切分資料集 (60% 訓練, 20% 驗證, 20% 測試) 
    # 第一次切分：切出 20% 測試集，剩下 80% 作為 訓練+驗證
    X1_train_val, X1_test, y1_train_val, y1_test = train_test_split(X_2d, y1, test_size=0.2, random_state=42, stratify=y1)
    X3_train_val, X3_test, y3_train_val, y3_test = train_test_split(X_2d, y3, test_size=0.2, random_state=42, stratify=y3)

    # 第二次切分：從 80% 中切出 25% 當驗證集 (0.8 * 0.25 = 0.2)，剩下的 75% 當訓練集 (0.8 * 0.75 = 0.6)
    X1_train, X1_val, y1_train, y1_val = train_test_split(X1_train_val, y1_train_val, test_size=0.25, random_state=42, stratify=y1_train_val)
    X3_train, X3_val, y3_train, y3_val = train_test_split(X3_train_val, y3_train_val, test_size=0.25, random_state=42, stratify=y3_train_val)

    # 5. 建立並訓練羅吉斯迴歸模型
    print("正在訓練羅吉斯迴歸線性探測器...")
    clf1 = LogisticRegression(random_state=42, max_iter=1000)
    clf2 = LogisticRegression(random_state=42, max_iter=1000)
    clf1.fit(X1_train, y1_train)
    clf2.fit(X3_train, y3_train)

    # 6. 進行預測並評估模型表現
    y1_pred = clf1.predict(X1_test)
    y1_proba = clf1.predict_proba(X1_test)[:, 1]  # 取得預測為 Unsafe (1) 的機率值
    y3_pred = clf2.predict(X3_test)
    y3_proba = clf2.predict_proba(X3_test)[:, 1]  # 取得預測為 Unsafe (1) 的機率值
    print("\n--- 🎯 模型預測結果報告 ---")
    print(f"y1 (Data Type) 的整體準確率 (Accuracy): {accuracy_score(y1_test, y1_pred):.2f}")
    print(f"y3 (Combined) 的整體準確率 (Accuracy): {accuracy_score(y3_test, y3_pred):.2f}")
    print("\n詳細分類報告:")
    print("\n--- y1 (Data Type) 分類報告 ---")
    print(classification_report(y1_test, y1_pred, target_names=['Safe (0)', 'Unsafe (1)']))
    print("\n--- y3 (Combined) 分類報告 ---")
    print(classification_report(y3_test, y3_pred, target_names=['Safe (0)', 'Unsafe (1)']))
    # 檢查過擬合
    print("檢查過擬合")
    print(f"訓練集y1(Data Type)準確率 (Accuracy): {clf1.score(X1_train, y1_train):.2f}")
    print(f"測試集y1(Data Type)準確率 (Accuracy): {clf1.score(X1_test, y1_test):.2f}")
    print(f"訓練集y3(Combined)準確率 (Accuracy): {clf2.score(X3_train, y3_train):.2f}")
    print(f"測試集y3(Combined)準確率 (Accuracy): {clf2.score(X3_test, y3_test):.2f}")
    print("\n==================== 分隔線 ====================\n")


    # 7. 儲存模型
    import joblib


    joblib.dump(clf1, f"lr_probe_model_y1_{i+1}.pkl")
    joblib.dump(clf2, f"lr_probe_model_y3_{i+1}.pkl")

    print(f"\n模型已儲存為: lr_probe_model_y1_{i+1}.pkl 和 lr_probe_model_y3_{i+1}.pkl")