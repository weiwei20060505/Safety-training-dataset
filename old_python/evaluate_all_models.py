import os
import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score, f1_score

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

IMG_DIR = "results/evaluation/images"
LOG_DIR = "results/evaluation"
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
sys.stdout = DualLogger(os.path.join(LOG_DIR, "evaluation_log.txt"))

# 1. 讀取實驗資料 (修正檔名讀取對應的檔案)
df = pd.read_pickle("experiment_results2.pkl")

# 2. 處理特徵矩陣 X
X_3d = np.array(df['hidden_state'].tolist())
num_layers = X_3d.shape[1]

for layer_idx in range(num_layers):
    print(f"\n==================== 評估第 {layer_idx+1} 層 ====================")
    X_2d = X_3d[:, layer_idx, :]  

    # 3. 處理目標變數 y
    y1 = df['data_type'].str.contains('harmful').astype(int) 
    y3 = (y1 == df['model_reply'].str.lower().str.contains('unsafe').astype(int)).astype(int)  

    # 4. 切分出與先前一致的測試集 (20%)
    _, X1_test, _, y1_test = train_test_split(X_2d, y1, test_size=0.2, random_state=42, stratify=y1)
    _, X3_test, _, y3_test = train_test_split(X_2d, y3, test_size=0.2, random_state=42, stratify=y3)

    # 5. 定義要載入的模型名稱與對應之正確分類路徑
    model_labels = ['SGD (Baseline)', 'MLP (Neural Network)', 'Random Forest', 'LightGBM']
    model_files_y1 = [
        f"results/sgd/models/sgd_probe_model_y1_{layer_idx+1}.pkl",
        f"results/mlp/models/mlp_probe_model_y1_{layer_idx+1}.pkl",
        f"results/random_forest/models/rf_probe_model_y1_{layer_idx+1}.pkl",
        f"results/lightgbm/models/lgb_probe_model_y1_{layer_idx+1}.pkl"
    ]
    model_files_y3 = [
        f"results/sgd/models/sgd_probe_model_y3_{layer_idx+1}.pkl",
        f"results/mlp/models/mlp_probe_model_y3_{layer_idx+1}.pkl",
        f"results/random_forest/models/rf_probe_model_y3_{layer_idx+1}.pkl",
        f"results/lightgbm/models/lgb_probe_model_y3_{layer_idx+1}.pkl"
    ]   
    # 6. 開始評估模型
    results_y1 = []
    print(f"=== 開始評估 Layer {layer_idx+1} 的 y1 模型 ===")
    for label, file_path in zip(model_labels, model_files_y1):
        try:
            model = joblib.load(file_path)
            y_pred = model.predict(X1_test)
            y_prob = model.predict_proba(X1_test)
            
            acc = accuracy_score(y1_test, y_pred)
            f1 = f1_score(y1_test, y_pred, average='macro')
            bce = log_loss(y1_test, y_prob, labels=[0, 1])
            
            results_y1.append({
                'Model': label,
                'Accuracy': acc,
                'F1-Score': f1,
                'BCE Loss': bce
            })
            print(f"[{label}] 載入成功並完成評估。")
        except FileNotFoundError:
            print(f"⚠️ 找不到模型檔案: {file_path}")
    results_y3 = []
    print(f"=== 開始評估 Layer {layer_idx+1} 的 y3 模型 ===")
    for label, file_path in zip(model_labels, model_files_y3):
        try:
            model = joblib.load(file_path)
            y_pred = model.predict(X3_test)
            y_prob = model.predict_proba(X3_test)

            acc = accuracy_score(y3_test, y_pred)
            f1 = f1_score(y3_test, y_pred, average='macro')
            bce = log_loss(y3_test, y_prob, labels=[0, 1])

            results_y3.append({
                'Model': label,
                'Accuracy': acc,
                'F1-Score': f1,
                'BCE Loss': bce
            })
            print(f"[{label}] 載入成功並完成評估。")
        except FileNotFoundError:
            print(f"⚠️ 找不到模型檔案: {file_path}")

    # 7. 顯示表格
    df_results_y1 = pd.DataFrame(results_y1)

    if not df_results_y1.empty:
        print("\n📊 y1 模型效能對比表：")
        print(df_results_y1.to_string(index=False))

        # 8. 繪製圖表
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        metrics = ['Accuracy', 'F1-Score', 'BCE Loss']
        colors = ['#4C72B0', '#55A868', '#C44E52']
        
        for ax, metric, color in zip(axes, metrics, colors):
            ax.bar(df_results_y1['Model'], df_results_y1[metric], color=color, alpha=0.8)
            ax.set_title(f'y1 Model Comparison ({metric})')
            ax.set_xticklabels(df_results_y1['Model'], rotation=15)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            for p in ax.patches:
                ax.annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', xytext=(0, 5), textcoords='offset points')

        plt.tight_layout()
        plt.savefig(os.path.join(IMG_DIR, f'model_benchmark_layer_{layer_idx+1}_y1.png'))
        plt.close()
        print(f"\n🎉 橫向對比圖表已儲存。")
    df_results_y3 = pd.DataFrame(results_y3)
    if not df_results_y3.empty:
        print("\n📊 y3 模型效能對比表：")
        print(df_results_y3.to_string(index=False))

        # 8. 繪製圖表
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        metrics = ['Accuracy', 'F1-Score', 'BCE Loss']
        colors = ['#4C72B0', '#55A868', '#C44E52']
        
        for ax, metric, color in zip(axes, metrics, colors):
            ax.bar(df_results_y3['Model'], df_results_y3[metric], color=color, alpha=0.8)
            ax.set_title(f'y3 Model Comparison ({metric})')
            ax.set_xticklabels(df_results_y3['Model'], rotation=15)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            for p in ax.patches:
                ax.annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', xytext=(0, 5), textcoords='offset points')

        plt.tight_layout()
        plt.savefig(os.path.join(IMG_DIR, f'model_benchmark_layer_{layer_idx+1}_y3.png'))
        plt.close()
        print(f"\n🎉 橫向對比圖表已儲存。")
    
    
    

