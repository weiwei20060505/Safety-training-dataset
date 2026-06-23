"""
統一機器學習模型訓練框架 (進階版：雙軌學習曲線監控)
================================================
目的：統一管理 5 種分類模型（SGD, LR, MLP, RF, LGB）的訓練和評估
進階特點：
  - [動態組] SGD, MLP, LGB: 支援 Epoch 逐輪驗證，防範過擬合
  - [靜態組] LR, RF: 支援切 5 份資料量驗證，評估資料需求
  - 自動生成專業對比與曲線圖表（支援中文 XY 軸與標題）
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb
import warnings

# 忽略不必要的警告
warnings.filterwarnings('ignore')

# ================= 第 1 區塊：日誌記錄器 =================
class DualLogger:
    """同時記錄到標準輸出和檔案"""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        try:
            self.terminal.write(message)
        except UnicodeEncodeError:
            encoding = getattr(self.terminal, 'encoding', 'utf-8') or 'utf-8'
            safe_msg = message.encode(encoding, errors='replace').decode(encoding)
            self.terminal.write(safe_msg)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# ================= 第 2 區塊：結果儲存類 =================
class ModelResults:
    """存儲單一模型的訓練結果與歷史數據"""
    def __init__(self, model_name):
        self.model_name = model_name
        
        # 最終評估指標
        self.y1_metrics = {}  
        self.y3_metrics = {}  
        self.y1_model = None  
        self.y3_model = None  
        
        # [動態組] 逐輪 (Epoch) 訓練歷史 (存放 dict: {'train_acc': [], 'val_acc': []})
        self.y1_epoch_history = None  
        self.y3_epoch_history = None  
        
        # [靜態組] 資料量 (Data-size) 學習曲線歷史 (存放 dict: {'sizes': [], 'train': [], 'val': []})
        self.y1_lc_history = None 
        self.y3_lc_history = None 

# ================= 第 3 區塊：數據加載和預處理 =================
class DataPreprocessor:
    """數據加載、特徵提取、目標變數生成"""
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        print("[1] 正在加載數據...")
        self.df = pd.read_pickle(self.data_path)
        print(f"  └─ 成功加載 {len(self.df)} 筆數據")
        return self.df

    def extract_features(self):
        print("[2] 正在提取隱藏狀態特徵...")
        self.X_3d = np.array(self.df['hidden_state'].tolist())
        print(f"  └─ 共 {self.X_3d.shape[1]} 層，每層 {self.X_3d.shape[2]} 維")
        return self.X_3d

    def create_targets(self):
        print("[3] 正在創建目標變數...")
        self.y1 = self.df['data_type'].str.contains('harmful').astype(int)
        self.y2 = self.df['model_reply'].str.lower().str.contains('unsafe').astype(int)
        self.y3 = (self.y1 == self.y2).astype(int)
        return self.y1, self.y3

# ================= 第 4 區塊：資料分割和標準化 =================
class DataSplitter:
    @staticmethod
    def split_and_scale(X, y, layer_idx, random_state=42):
        # 60% Train, 20% Val, 20% Test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.25, random_state=random_state, stratify=y_train_val
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler

# ================= 第 5 區塊：模型訓練管理器 =================
class UnifiedModelTrainer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def train_sgd(self, X_train, X_val, X_test, y_train, y_val, y_test, y_name):
        """[動態組] SGD 逐輪訓練 (Epochs)"""
        print(f"\n  [SGD] 訓練 {y_name} 模型 (逐輪監控)...")
        y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
        
        # 🌟【關鍵修改 1】在訓練前，先取得所有可能的類別標籤 (例如 [0, 1])
        classes = np.unique(y_train_np)
        
        clf = SGDClassifier(
            loss='log_loss', 
            penalty='l2', 
            alpha=0.01, 
            learning_rate='adaptive', 
            eta0=0.01, 
            class_weight=None, 
            random_state=42
        )
        
        epochs = 50
        batch_size = 64
        history = {'sizes': [], 'train_acc': [], 'val_acc': []}

        for epoch in range(epochs):
            # 每個 Epoch 開始前打亂訓練資料
            indices = np.random.permutation(len(X_train))
            X_shuffled = X_train[indices]
            y_shuffled = y_train_np[indices]
            
            # Mini-batch 訓練
            for start_idx in range(0, len(X_train), batch_size):
                end_idx = min(start_idx + batch_size, len(X_train))
                
                # 🌟【關鍵修改 2】將 classes 參數傳遞給 partial_fit
                clf.partial_fit(
                    X_shuffled[start_idx:end_idx], 
                    y_shuffled[start_idx:end_idx], 
                    classes=classes
                )
                
            history['sizes'].append(epoch + 1)
            history['train_acc'].append(accuracy_score(y_train, clf.predict(X_train)))
            history['val_acc'].append(accuracy_score(y_val, clf.predict(X_val)))

        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        
        return clf, y_pred, y_pred_proba, history, None

    def train_mlp(self, X_train, X_val, X_test, y_train, y_val, y_test, y_name):
        """[動態組] MLP 逐輪訓練 (Epochs)"""
        print(f"\n  [MLP] 訓練 {y_name} 模型 (逐輪監控)...")
        y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
        clf = MLPClassifier(hidden_layer_sizes=(32,), random_state=42,alpha=0.01)
        classes = np.unique(y_train_np)
        epochs = 100
        batch_size = 64
        history = {'sizes': [], 'train_acc': [], 'val_acc': []}

        for epoch in range(epochs):
            # 每個 Epoch 開始前打亂訓練資料
            indices = np.random.permutation(len(X_train))
            X_shuffled = X_train[indices]
            y_shuffled = y_train_np[indices]
            
            # Mini-batch 訓練
            for start_idx in range(0, len(X_train), batch_size):
                end_idx = min(start_idx + batch_size, len(X_train))
                clf.partial_fit(X_shuffled[start_idx:end_idx], y_shuffled[start_idx:end_idx], classes=classes)
                
            history['sizes'].append(epoch + 1)
            history['train_acc'].append(accuracy_score(y_train, clf.predict(X_train)))
            history['val_acc'].append(accuracy_score(y_val, clf.predict(X_val)))

        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        return clf, y_pred, y_pred_proba, history, None

    def train_lgb(self, X_train, X_val, X_test, y_train, y_val, y_test, y_name):
        """[動態組] LightGBM 逐樹訓練 (Trees)"""
        print(f"\n  [LGB] 訓練 {y_name} 模型 (逐樹監控)...")
        clf = lgb.LGBMClassifier(
                n_estimators=100, 
                learning_rate=0.05, 
                random_state=42, 
                max_depth=3,          # 限制樹深
                num_leaves=7,         # 限制葉子數
                reg_alpha=0.1,        # L1 正則化
                reg_lambda=0.1,       # L2 正則化
                verbose=-1
            )
        clf.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_names=['train', 'val'],
            eval_metric='binary_error',
            callbacks=[lgb.log_evaluation(period=0)]
        )

        history = {
            'sizes': list(range(1, len(clf.evals_result_['train']['binary_error']) + 1)),
            'train_acc': [1 - x for x in clf.evals_result_['train']['binary_error']],
            'val_acc': [1 - x for x in clf.evals_result_['val']['binary_error']]
        }

        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        return clf, y_pred, y_pred_proba, history, None

    def train_lr(self, X_train, X_val, X_test, y_train, y_val, y_test, y_name):
        """[靜態組] LR 資料量學習曲線 (學習曲線函數)"""
        print(f"\n  [LR] 訓練 {y_name} 模型 (評估 5 份資料量)...")
        clf = LogisticRegression(C=0.01, penalty='l2', max_iter=1000, random_state=42)

        # 使用 Scikit-learn 的 learning_curve 函數，自動處理分層 Stratified 5-Fold 與累積增長
        train_sizes, train_scores, val_scores = learning_curve(
            clf, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1,
            train_sizes=np.linspace(0.2, 1.0, 5) # 20%, 40%, 60%, 80%, 100%
        )
        lc_history = {
            'sizes': train_sizes,
            'train': train_scores.mean(axis=1),
            'val': val_scores.mean(axis=1)
        }

        # 最終模型採用完整訓練集
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        return clf, y_pred, y_pred_proba, None, lc_history

    def train_rf(self, X_train, X_val, X_test, y_train, y_val, y_test, y_name):
        """[靜態組] RF 資料量學習曲線 (學習曲線函數)"""
        print(f"\n  [RF] 訓練 {y_name} 模型 (評估 5 份資料量)...")
        clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42, n_jobs=-1,min_samples_leaf=10)
        
        # 使用 Scikit-learn 的 learning_curve 函數，自動處理分層 Stratified 5-Fold 與累積增長
        train_sizes, train_scores, val_scores = learning_curve(
            clf, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1,
            train_sizes=np.linspace(0.2, 1.0, 5) # 20%, 40%, 60%, 80%, 100%
        )
        lc_history = {
            'sizes': train_sizes,
            'train': train_scores.mean(axis=1),
            'val': val_scores.mean(axis=1)
        }

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        return clf, y_pred, y_pred_proba, None, lc_history

# ================= 第 6 區塊：評估指標 =================
class MetricsCalculator:
    @staticmethod
    def calculate_metrics(y_true, y_pred, y_pred_proba):
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }
        try:
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
        except:
            metrics['auc'] = np.nan
        return metrics

# ================= 第 7 區塊：繪圖工具 (包含中文字體設定) =================
class PlotGenerator:
    @staticmethod
    def setup_chinese_font():
        """設定 Matplotlib 支援中文顯示"""
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'PMingLiU', 'DFKai-SB', 'DejaVu Sans', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False  # 解決負號變方塊的問題

    @staticmethod
    def plot_all_curves(all_results, output_dir, target_name, layer_idx):
        """繪製 Epoch/Tree 曲線與 Data-size 曲線"""
        PlotGenerator.setup_chinese_font()
        models = all_results.keys()
        
        # --- 圖 1：動態組 (SGD, MLP 逐輪訓練學習曲線，LGB 逐樹訓練學習曲線) ---
        epoch_models = [m for m in models if getattr(all_results[m], f'{target_name}_epoch_history') is not None]
        if epoch_models:
            fig, axes = plt.subplots(1, len(epoch_models), figsize=(6 * len(epoch_models), 5))
            if len(epoch_models) == 1: axes = [axes]
            
            fig.suptitle(f'第 {layer_idx+1} 層 - {target_name.upper()} 任務動態組訓練學習曲線', fontsize=16, fontweight='bold')
            
            for ax, model_name in zip(axes, epoch_models):
                history = getattr(all_results[model_name], f'{target_name}_epoch_history')
                sizes = history['sizes']
                
                ax.plot(sizes, history['train_acc'], label='訓練集 (Train)', color='blue', linewidth=2)
                ax.plot(sizes, history['val_acc'], label='驗證集 (Val)', color='orange', linewidth=2)
                
                if model_name == 'LGB':
                    ax.set_title(f'{model_name} 逐樹收斂狀態', fontsize=14)
                    ax.set_xlabel('樹的數量 (Number of Trees)', fontsize=12)
                else:
                    ax.set_title(f'{model_name} 逐輪收斂狀態', fontsize=14)
                    ax.set_xlabel('訓練輪數 (Epochs)', fontsize=12)
                    
                ax.set_ylabel('準確率 (Accuracy)', fontsize=12)
                ax.legend(fontsize=10)
                ax.grid(True, linestyle='--', alpha=0.6)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'learning_curves_step_{target_name}.png'), dpi=150)
            plt.close()

        # --- 圖 2：靜態組 (LR, RF 資料量需求曲線) ---
        lc_models = [m for m in models if getattr(all_results[m], f'{target_name}_lc_history') is not None]
        if lc_models:
            fig, axes = plt.subplots(1, len(lc_models), figsize=(6 * len(lc_models), 5))
            if len(lc_models) == 1: axes = [axes]
            
            fig.suptitle(f'第 {layer_idx+1} 層 - {target_name.upper()} 任務資料量需求曲線 (5 份分割)', fontsize=16, fontweight='bold')
            
            for ax, model_name in zip(axes, lc_models):
                history = getattr(all_results[model_name], f'{target_name}_lc_history')
                sizes = history['sizes']
                
                ax.plot(sizes, history['train'], 'o-', label='訓練集 (Train)', color='green', linewidth=2)
                ax.plot(sizes, history['val'], 'o-', label='交叉驗證集 (CV)', color='red', linewidth=2)
                
                ax.set_title(f'{model_name} 資料量與極限關係', fontsize=14)
                ax.set_xlabel('訓練樣本數 (Data Size)', fontsize=12)
                ax.set_ylabel('準確率 (Accuracy)', fontsize=12)
                ax.legend(fontsize=10)
                ax.grid(True, linestyle='--', alpha=0.6)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'learning_curves_data_{target_name}.png'), dpi=150)
            plt.close()

    @staticmethod
    def plot_model_comparison(all_results, output_dir, target_name, layer_idx):
        """繪製各模型在測試集上的性能對比條形圖 (5 個指標)"""
        PlotGenerator.setup_chinese_font()
        models = list(all_results.keys())
        
        # 提取指標
        metrics_names = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        display_names = {
            'accuracy': '準確率 (Accuracy)',
            'precision': '精確率 (Precision)',
            'recall': '召回率 (Recall)',
            'f1': 'F1 值 (F1 Score)',
            'auc': 'AUC 值 (ROC AUC)'
        }
        
        data = {m: [] for m in models}
        for m in models:
            metrics = getattr(all_results[m], f'{target_name}_metrics')
            for name in metrics_names:
                data[m].append(metrics.get(name, np.nan))
                
        # 繪圖
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        fig.suptitle(f'第 {layer_idx+1} 層 - {target_name.upper()} 任務模型性能對比', fontsize=16, fontweight='bold')
        
        colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974']
        
        for idx, metric_name in enumerate(metrics_names):
            ax = axes[idx]
            scores = [data[m][idx] for m in models]
            
            bars = ax.bar(models, scores, color=colors, alpha=0.85, edgecolor='black', linewidth=0.7)
            ax.set_title(display_names[metric_name], fontsize=13, fontweight='bold')
            ax.set_ylim(0, 1.1)
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            
            # 在條形圖上方顯示數值
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax.annotate(f'{height:.3f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=9, fontweight='bold')
                                
        plt.tight_layout()
        filename = f'model_comparison_layer_{layer_idx+1}_{target_name}.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=150)
        plt.close()

# ================= 第 8 區塊：主程式 =================
def main():
    print("\n" + "="*80)
    print("雙軌機器學習模型訓練框架 - 開始執行")
    print("="*80)

    # 請確保這裡的檔案名稱與你的環境相符
    DATA_PATH = "experiment_results_train_1000.pkl"
    if not os.path.exists(DATA_PATH):
        DATA_PATH = "experiment_results.pkl"
    if not os.path.exists(DATA_PATH):
        print(f"錯誤: 找不到數據檔案。請確保 {DATA_PATH} 存在。")
        sys.exit(1)

    OUTPUT_DIR = "results/unified_training"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sys.stdout = DualLogger(os.path.join(OUTPUT_DIR, "training_log.txt"))

    # 讀取資料
    preprocessor = DataPreprocessor(DATA_PATH)
    df = preprocessor.load_data()
    X_3d = preprocessor.extract_features()
    y1, y3 = preprocessor.create_targets()

    num_layers = X_3d.shape[1]
    
    for layer_idx in range(num_layers):
        print(f"\n" + "="*60)
        print(f"[開始訓練第 {layer_idx + 1} / {num_layers} 層特徵]")
        print("="*60)
        
        X_2d = X_3d[:, layer_idx, :]

        # 資料分割
        X_train_y1, X_val_y1, X_test_y1, y_train_y1, y_val_y1, y_test_y1, _ = DataSplitter.split_and_scale(X_2d, y1, layer_idx)
        X_train_y3, X_val_y3, X_test_y3, y_train_y3, y_val_y3, y_test_y3, _ = DataSplitter.split_and_scale(X_2d, y3, layer_idx)

        layer_output_dir = os.path.join(OUTPUT_DIR, f"layer_{layer_idx+1}")
        trainer = UnifiedModelTrainer(output_dir=layer_output_dir)
        
        models_to_train = [
            ('SGD', trainer.train_sgd),
            ('MLP', trainer.train_mlp),
            ('LGB', trainer.train_lgb),
            ('LR', trainer.train_lr),
            ('RF', trainer.train_rf)
        ]

        all_results = {}

        for model_name, train_func in models_to_train:
            results = ModelResults(model_name)
            
            # 訓練 y1
            clf_y1, y1_pred, y1_proba, epoch_hist_y1, lc_hist_y1 = train_func(
                X_train_y1, X_val_y1, X_test_y1, y_train_y1, y_val_y1, y_test_y1, 'Y1'
            )
            results.y1_epoch_history = epoch_hist_y1
            results.y1_lc_history = lc_hist_y1
            results.y1_metrics = MetricsCalculator.calculate_metrics(y_test_y1, y1_pred, y1_proba)
            
            # 保存 y1 模型
            joblib.dump(clf_y1, os.path.join(layer_output_dir, f"{model_name.lower()}_y1.pkl"))
            
            # 訓練 y3
            clf_y3, y3_pred, y3_proba, epoch_hist_y3, lc_hist_y3 = train_func(
                X_train_y3, X_val_y3, X_test_y3, y_train_y3, y_val_y3, y_test_y3, 'Y3'
            )
            results.y3_epoch_history = epoch_hist_y3
            results.y3_lc_history = lc_hist_y3
            results.y3_metrics = MetricsCalculator.calculate_metrics(y_test_y3, y3_pred, y3_proba)
            
            # 保存 y3 模型
            joblib.dump(clf_y3, os.path.join(layer_output_dir, f"{model_name.lower()}_y3.pkl"))
            
            all_results[model_name] = results

        # 輸出指標表格
        print(f"\n📊 [第 {layer_idx+1} 層] Y1 (有害性任務) 性能指標:")
        metrics_df_y1 = pd.DataFrame({m: all_results[m].y1_metrics for m in all_results}).T
        print(metrics_df_y1.to_string())

        print(f"\n📊 [第 {layer_idx+1} 層] Y3 (一致性任務) 性能指標:")
        metrics_df_y3 = pd.DataFrame({m: all_results[m].y3_metrics for m in all_results}).T
        print(metrics_df_y3.to_string())

        # 生成進階曲線與對比圖表
        print(f"\n[生成圖表] 正在繪製第 {layer_idx+1} 層的學習曲線與模型對比圖...")
        PlotGenerator.plot_all_curves(all_results, layer_output_dir, 'y1', layer_idx)
        PlotGenerator.plot_all_curves(all_results, layer_output_dir, 'y3', layer_idx)
        PlotGenerator.plot_model_comparison(all_results, layer_output_dir, 'y1', layer_idx)
        PlotGenerator.plot_model_comparison(all_results, layer_output_dir, 'y3', layer_idx)
        
    print(f"\n[OK] 所有 6 層特徵的模型訓練、評估與繪圖已全部完成！")
    print(f"結果與圖表已儲存至 {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
