"""
統一機器學習模型訓練框架 (進階版：雙軌學習曲線監控)
================================================
目的：統一管理 5 種分類模型（SGD, LR, MLP, RF, LGB）的訓練 and 評估
進階特點：
  - [動態組] SGD, MLP, LGB: 支援 Epoch 逐輪驗證，防範過擬合
  - [靜態組] LR, RF: 支援切 5 份資料量驗證，評估資料需求
  - 自動生成專業對比與曲線圖表（支援中文 XY 軸與標題）
"""

from genericpath import exists
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
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    log_loss, roc_curve, auc, balanced_accuracy_score
)
import copy
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
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
        self.y2_metrics = {}  
        self.y3_metrics = {}  
        self.y1_model = None  
        self.y2_model = None  
        self.y3_model = None  
        
        # [動態組] 逐輪 (Epoch) 訓練歷史
        self.y1_epoch_history = None  
        self.y2_epoch_history = None  
        self.y3_epoch_history = None  
        
        # [靜態組] 資料量 (Data-size) 學習曲線歷史
        self.y1_lc_history = None 
        self.y2_lc_history = None 
        self.y3_lc_history = None 

        # [靜態組] 資料量 (Data-size) Balanced Accuracy 學習曲線歷史
        self.y1_bal_lc_history = None
        self.y2_bal_lc_history = None
        self.y3_bal_lc_history = None
        
        # [ROC 曲線相關] 儲存測試集的真實標籤與預測機率
        self.y1_test_true = None
        self.y1_test_proba = None
        self.y2_test_true = None
        self.y2_test_proba = None
        self.y3_test_true = None
        self.y3_test_proba = None

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
        self.y1 = self.df['model_reply'].str.lower().str.contains('unsafe').astype(int)
        self.y2 = self.df['data_type'].str.contains('harmful').astype(int)
        self.y3 = (self.y1 == self.y2).astype(int)
        return self.y1, self.y2, self.y3

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
        return X_train, X_val, X_test, y_train, y_val, y_test, None

# ================= 第 5 區塊：模型訓練管理器 =================
class UnifiedModelTrainer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def train_sgd(self, X_train, X_val, X_test, y_train, y_val, y_test, y_name):
        """[動態組] SGD 逐輪訓練 (Epochs)"""
        print(f"\n  [SGD] 訓練 {y_name} 模型 (逐輪監控)...")
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        sampler = RandomUnderSampler(random_state=42)
        X_train_res, y_train_res = sampler.fit_resample(X_train_scaled, y_train)
        
        pca = PCA(n_components=128, random_state=42)
        X_train_pca = pca.fit_transform(X_train_res)
        X_val_pca = pca.transform(X_val_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        classes = np.unique(y_train_res)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train_res)
        class_weight_dict = {classes[j]: class_weights[j] for j in range(len(classes))}
        
        clf = SGDClassifier(
            loss='log_loss', 
            penalty='l2', 
            alpha=0.01, 
            learning_rate='adaptive', 
            eta0=0.0001, 
            class_weight=class_weight_dict, 
            random_state=42
        )
        
        epochs = 100
        batch_size = 64
        history = {'sizes': [], 'train_acc': [], 'val_acc': [], 'train_bal_acc': [], 'val_bal_acc': [], 'train_loss': [], 'val_loss': []}

        # 用於追蹤最佳模型
        best_val_loss = float('inf')
        best_clf = None
        best_epoch = 1

        for epoch in range(epochs):
            indices = np.random.permutation(len(X_train_pca))
            X_shuffled = X_train_pca[indices]
            y_shuffled = y_train_res[indices]
            
            for start_idx in range(0, len(X_train_pca), batch_size):
                end_idx = min(start_idx + batch_size, len(X_train_pca))
                clf.partial_fit(X_shuffled[start_idx:end_idx], y_shuffled[start_idx:end_idx], classes=classes)
                
            train_preds = clf.predict(X_train_pca)
            val_preds = clf.predict(X_val_pca)
            train_acc = accuracy_score(y_train_res, train_preds)
            val_acc = accuracy_score(y_val, val_preds)
            train_bal_acc = balanced_accuracy_score(y_train_res, train_preds)
            val_bal_acc = balanced_accuracy_score(y_val, val_preds)

            train_proba = clf.predict_proba(X_train_pca)
            val_proba = clf.predict_proba(X_val_pca)
            train_loss = log_loss(y_train_res, train_proba, labels=classes)
            val_loss = log_loss(y_val, val_proba, labels=classes)

            history['sizes'].append(epoch + 1)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['train_bal_acc'].append(train_bal_acc)
            history['val_bal_acc'].append(val_bal_acc)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            # 記錄最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_clf = copy.deepcopy(clf)
                best_epoch = epoch + 1
                
        # 訓練完成後，此時的 clf 即為 last_clf
        last_clf = copy.deepcopy(clf)
        if best_clf is None:
            best_clf = last_clf
            best_epoch = epochs
            
        history['best_epoch'] = best_epoch

        y_pred = best_clf.predict(X_test_pca)
        y_pred_proba = best_clf.predict_proba(X_test_pca)[:, 1]
        
        final_pipeline_best = ImbPipeline([
            ('scaler', scaler),
            ('pca', pca),
            ('clf', best_clf)
        ])
        final_pipeline_last = ImbPipeline([
            ('scaler', scaler),
            ('pca', pca),
            ('clf', last_clf)
        ])
        return final_pipeline_best, final_pipeline_last, y_pred, y_pred_proba, history, None, None

    def train_mlp(self, X_train, X_val, X_test, y_train, y_val, y_test, y_name):
        """[動態組] MLP 逐輪訓練 (Epochs)"""
        print(f"\n  [MLP] 訓練 {y_name} 模型 (逐輪監控)...")
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        sampler = RandomUnderSampler(random_state=42)
        X_train_res, y_train_res = sampler.fit_resample(X_train_scaled, y_train)
        
        pca = PCA(n_components=128, random_state=42)
        X_train_pca = pca.fit_transform(X_train_res)
        X_val_pca = pca.transform(X_val_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        clf = MLPClassifier(hidden_layer_sizes=(128,), random_state=42, alpha=0.01)
        classes = np.unique(y_train_res)
        epochs = 100
        batch_size = 64
        history = {'sizes': [], 'train_acc': [], 'val_acc': [], 'train_bal_acc': [], 'val_bal_acc': [], 'train_loss': [], 'val_loss': []}

        # 用於追蹤最佳模型
        best_val_loss = float('inf')
        best_clf = None
        best_epoch = 1

        for epoch in range(epochs):
            indices = np.random.permutation(len(X_train_pca))
            X_shuffled = X_train_pca[indices]
            y_shuffled = y_train_res[indices]
            
            for start_idx in range(0, len(X_train_pca), batch_size):
                end_idx = min(start_idx + batch_size, len(X_train_pca))
                clf.partial_fit(X_shuffled[start_idx:end_idx], y_shuffled[start_idx:end_idx], classes=classes)
                
            train_preds = clf.predict(X_train_pca)
            val_preds = clf.predict(X_val_pca)
            train_acc = accuracy_score(y_train_res, train_preds)
            val_acc = accuracy_score(y_val, val_preds)
            train_bal_acc = balanced_accuracy_score(y_train_res, train_preds)
            val_bal_acc = balanced_accuracy_score(y_val, val_preds)

            train_proba = clf.predict_proba(X_train_pca)
            val_proba = clf.predict_proba(X_val_pca)
            train_loss = log_loss(y_train_res, train_proba, labels=classes)
            val_loss = log_loss(y_val, val_proba, labels=classes)

            history['sizes'].append(epoch + 1)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['train_bal_acc'].append(train_bal_acc)
            history['val_bal_acc'].append(val_bal_acc)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            # 記錄最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_clf = copy.deepcopy(clf)
                best_epoch = epoch + 1
                
        # 訓練完成後，此時的 clf 即為 last_clf
        last_clf = copy.deepcopy(clf)
        if best_clf is None:
            best_clf = last_clf
            best_epoch = epochs
            
        history['best_epoch'] = best_epoch

        y_pred = best_clf.predict(X_test_pca)
        y_pred_proba = best_clf.predict_proba(X_test_pca)[:, 1]
        
        final_pipeline_best = ImbPipeline([
            ('scaler', scaler),
            ('pca', pca),
            ('clf', best_clf)
        ])
        final_pipeline_last = ImbPipeline([
            ('scaler', scaler),
            ('pca', pca),
            ('clf', last_clf)
        ])
        return final_pipeline_best, final_pipeline_last, y_pred, y_pred_proba, history, None, None

    def train_lgb(self, X_train, X_val, X_test, y_train, y_val, y_test, y_name):
        """[動態組] LightGBM 逐樹訓練 (Trees)"""
        print(f"\n  [LGB] 訓練 {y_name} 模型 (逐樹監控)...")
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        sampler = RandomUnderSampler(random_state=42)
        X_train_res, y_train_res = sampler.fit_resample(X_train_scaled, y_train)
        
        pca = PCA(n_components=128, random_state=42)
        X_train_pca = pca.fit_transform(X_train_res)
        X_val_pca = pca.transform(X_val_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        clf = lgb.LGBMClassifier(
            n_estimators=100, 
            learning_rate=0.05, 
            random_state=42, 
            max_depth=10,
            num_leaves=31,
            reg_alpha=0.05,
            reg_lambda=0.05,
            verbose=-1
        )
        
        def lgb_balanced_accuracy(y_true, y_pred):
            y_pred_binary = (y_pred > 0.5).astype(int)
            score = balanced_accuracy_score(y_true, y_pred_binary)
            return 'balanced_accuracy', score, True
        
        # 利用 eval_metric 紀錄訓練過程的 binary_error (錯誤率) 與 binary_logloss，跑滿 100 棵樹
        clf.fit(
            X_train_pca, y_train_res,
            eval_set=[(X_train_pca, y_train_res), (X_val_pca, y_val)],
            eval_names=['train', 'val'],
            eval_metric=['binary_error', 'binary_logloss', lgb_balanced_accuracy],
            callbacks=[
                lgb.log_evaluation(period=0)
            ]
        )

        history = {
            'sizes': list(range(1, len(clf.evals_result_['train']['binary_error']) + 1)),
            'train_acc': [1 - x for x in clf.evals_result_['train']['binary_error']],
            'val_acc': [1 - x for x in clf.evals_result_['val']['binary_error']],
            'train_bal_acc': clf.evals_result_['train']['balanced_accuracy'],
            'val_bal_acc': clf.evals_result_['val']['balanced_accuracy'],
            'train_loss': clf.evals_result_['train']['binary_logloss'],
            'val_loss': clf.evals_result_['val']['binary_logloss']
        }

        # 尋找驗證集上 Loss 最低的最佳樹數量 (argmin)
        val_losses = clf.evals_result_['val']['binary_logloss']
        best_iteration = int(np.argmin(val_losses) + 1)
        print(f"    [LGB] 訓練完成，最佳樹數量 (best_iteration): {best_iteration} (最佳驗證 Loss: {val_losses[best_iteration-1]:.4f})")
        
        history['best_epoch'] = best_iteration

        # 構建 best 模型：複製一個 clf，手動設定最佳輪數
        best_clf = copy.deepcopy(clf)
        best_clf._best_iteration = best_iteration
        
        # last 模型：複製一個 clf，不使用 best_iteration_
        last_clf = copy.deepcopy(clf)
        last_clf._best_iteration = 0

        # 預測使用 best_clf
        y_pred = best_clf.predict(X_test_pca)
        y_pred_proba = best_clf.predict_proba(X_test_pca)[:, 1]
        
        final_pipeline_best = ImbPipeline([
            ('scaler', scaler),
            ('pca', pca),
            ('clf', best_clf)
        ])
        final_pipeline_last = ImbPipeline([
            ('scaler', scaler),
            ('pca', pca),
            ('clf', last_clf)
        ])
        return final_pipeline_best, final_pipeline_last, y_pred, y_pred_proba, history, None, None

    def train_lr(self, X_train, X_val, X_test, y_train, y_val, y_test, y_name):
        """[靜態組] LR 資料量學習曲線 (學習曲線函數)"""
        print(f"\n  [LR] 訓練 {y_name} 模型 (評估 5 份資料量)...")
        
        pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('sampler', RandomUnderSampler(random_state=42)),
            ('pca', PCA(n_components=128, random_state=42)),
            ('clf', LogisticRegression(C=0.01, penalty='l2', max_iter=1000, random_state=42))
        ])

        train_sizes, train_scores, val_scores = learning_curve(
            pipeline, X_train, y_train, cv=5, scoring='accuracy', n_jobs=2,
            train_sizes=np.linspace(0.2, 1.0, 5)
        )
        lc_history = {
            'sizes': train_sizes,
            'train': train_scores.mean(axis=1),
            'val': val_scores.mean(axis=1)
        }

        train_sizes, train_scores_bal, val_scores_bal = learning_curve(
            pipeline, X_train, y_train, cv=5, scoring='balanced_accuracy', n_jobs=2,
            train_sizes=np.linspace(0.2, 1.0, 5)
        )
        bal_lc_history = {
            'sizes': train_sizes,
            'train': train_scores_bal.mean(axis=1),
            'val': val_scores_bal.mean(axis=1)
        }

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        return pipeline, pipeline, y_pred, y_pred_proba, None, lc_history, bal_lc_history

    def train_rf(self, X_train, X_val, X_test, y_train, y_val, y_test, y_name):
        """[靜態組] RF 資料量學習曲線 (學習曲線函數)"""
        print(f"\n  [RF] 訓練 {y_name} 模型 (評估 5 份資料量)...")
        
        pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('sampler', RandomUnderSampler(random_state=42)),
            ('pca', PCA(n_components=128, random_state=42)),
            ('clf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=2))
        ])
        
        train_sizes, train_scores, val_scores = learning_curve(
            pipeline, X_train, y_train, cv=5, scoring='accuracy', n_jobs=2,
            train_sizes=np.linspace(0.2, 1.0, 5)
        )
        lc_history = {
            'sizes': train_sizes,
            'train': train_scores.mean(axis=1),
            'val': val_scores.mean(axis=1)
        }

        train_sizes, train_scores_bal, val_scores_bal = learning_curve(
            pipeline, X_train, y_train, cv=5, scoring='balanced_accuracy', n_jobs=2,
            train_sizes=np.linspace(0.2, 1.0, 5)
        )
        bal_lc_history = {
            'sizes': train_sizes,
            'train': train_scores_bal.mean(axis=1),
            'val': val_scores_bal.mean(axis=1)
        }

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        return pipeline, pipeline, y_pred, y_pred_proba, None, lc_history, bal_lc_history

# ================= 第 6 區塊：評估指標 =================
class MetricsCalculator:
    @staticmethod
    def calculate_metrics(y_true, y_pred, y_pred_proba):
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
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
            fig, axes = plt.subplots(2, len(epoch_models), figsize=(6 * len(epoch_models), 9))
            if len(epoch_models) == 1:
                axes = np.expand_dims(axes, axis=1)
            
            fig.suptitle(f'第 {layer_idx+1} 層 - {target_name.upper()} 任務動態組訓練學習曲線 (含早停監控)', fontsize=16, fontweight='bold')
            
            for idx, model_name in enumerate(epoch_models):
                history = getattr(all_results[model_name], f'{target_name}_epoch_history')
                sizes = history['sizes']
                best_epoch = history.get('best_epoch', None)
                
                # 上排：Accuracy
                ax_acc = axes[0, idx]
                ax_acc.plot(sizes, history['train_acc'], label='訓練集 (Train Acc)', color='blue', linewidth=2)
                ax_acc.plot(sizes, history['val_acc'], label='驗證集 (Val Acc)', color='orange', linewidth=2)
                if best_epoch is not None:
                    ax_acc.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'最佳輪數 ({best_epoch})')
                
                # 下排：Loss
                ax_loss = axes[1, idx]
                ax_loss.plot(sizes, history['train_loss'], label='訓練集 (Train Loss)', color='cyan', linewidth=2)
                ax_loss.plot(sizes, history['val_loss'], label='驗證集 (Val Loss)', color='red', linewidth=2)
                if best_epoch is not None:
                    ax_loss.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'最佳輪數 ({best_epoch})')
                
                if model_name == 'LGB':
                    ax_acc.set_title(f'{model_name} 逐樹收斂 - Accuracy', fontsize=13)
                    ax_loss.set_title(f'{model_name} 逐樹收斂 - Loss', fontsize=13)
                    ax_loss.set_xlabel('樹的數量 (Number of Trees)', fontsize=12)
                else:
                    ax_acc.set_title(f'{model_name} 逐輪收斂 - Accuracy', fontsize=13)
                    ax_loss.set_title(f'{model_name} 逐輪收斂 - Loss', fontsize=13)
                    ax_loss.set_xlabel('訓練輪數 (Epochs)', fontsize=12)
                    
                ax_acc.set_ylabel('準確率 (Accuracy)', fontsize=12)
                ax_loss.set_ylabel('對數損失 (Log Loss)', fontsize=12)
                ax_acc.legend(fontsize=10)
                ax_loss.legend(fontsize=10)
                ax_acc.grid(True, linestyle='--', alpha=0.6)
                ax_loss.grid(True, linestyle='--', alpha=0.6)
            
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
    def plot_roc_curve(all_results, output_dir, target_name, layer_idx):
        """繪製所有模型的 ROC 曲線對比圖"""
        PlotGenerator.setup_chinese_font()
        models = list(all_results.keys())
        
        plt.figure(figsize=(8, 7))
        
        # 繪製隨機猜測線
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5, label='隨機猜測 (AUC = 0.500)')
        
        colors = {
            'SGD': '#1f77b4',
            'MLP': '#ff7f0e',
            'LGB': '#2ca02c',
            'LR': '#d62728',
            'RF': '#9467bd'
        }
        
        for model_name in models:
            results = all_results[model_name]
            y_true = getattr(results, f'{target_name}_test_true')
            y_proba = getattr(results, f'{target_name}_test_proba')
            
            if y_true is not None and y_proba is not None:
                # 排除 NaN
                if np.isnan(y_proba).any() or np.isnan(y_true).any():
                    continue
                
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                auc_score = auc(fpr, tpr)
                color = colors.get(model_name, '#7f7f7f')
                plt.plot(fpr, tpr, color=color, linewidth=2, 
                         label=f'{model_name} (AUC = {auc_score:.3f})')
        
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel('偽陽性率 (False Positive Rate)', fontsize=12)
        plt.ylabel('真陽性率 (True Positive Rate)', fontsize=12)
        plt.title(f'第 {layer_idx+1} 層 - {target_name.upper()} 任務模型 ROC 曲線對比', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        filename = f'model_roc_curve_layer_{layer_idx+1}_{target_name}.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=150)
        plt.close()

    @staticmethod
    def plot_balanced_accuracy_curves(all_results, output_dir, target_name, layer_idx):
        """繪製 Epoch/Tree 以及 Data-size 的 Balanced Accuracy 學習曲線"""
        PlotGenerator.setup_chinese_font()
        models = all_results.keys()
        
        # 1. 動態組的 Balanced Accuracy 曲線
        epoch_models = [m for m in models if getattr(all_results[m], f'{target_name}_epoch_history') is not None]
        if epoch_models:
            fig, axes = plt.subplots(1, len(epoch_models), figsize=(6 * len(epoch_models), 5))
            if len(epoch_models) == 1: axes = [axes]
            
            fig.suptitle(f'第 {layer_idx+1} 層 - {target_name.upper()} 任務動態組 Balanced Accuracy 學習曲線', fontsize=16, fontweight='bold')
            
            for ax, model_name in zip(axes, epoch_models):
                history = getattr(all_results[model_name], f'{target_name}_epoch_history')
                sizes = history['sizes']
                best_epoch = history.get('best_epoch', None)
                
                # 繪製 Balanced Accuracy 曲線
                ax.plot(sizes, history['train_bal_acc'], label='訓練集 (Train Bal Acc)', color='blue', linewidth=2)
                ax.plot(sizes, history['val_bal_acc'], label='驗證集 (Val Bal Acc)', color='orange', linewidth=2)
                if best_epoch is not None:
                    ax.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'最佳輪數 ({best_epoch})')
                
                if model_name == 'LGB':
                    ax.set_title(f'{model_name} 逐樹收斂 - Balanced Acc', fontsize=14)
                    ax.set_xlabel('樹的數量 (Number of Trees)', fontsize=12)
                else:
                    ax.set_title(f'{model_name} 逐輪收斂 - Balanced Acc', fontsize=14)
                    ax.set_xlabel('訓練輪數 (Epochs)', fontsize=12)
                    
                ax.set_ylabel('平衡準確率 (Balanced Accuracy)', fontsize=12)
                ax.legend(fontsize=10)
                ax.grid(True, linestyle='--', alpha=0.6)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'learning_curves_step_bal_acc_{target_name}.png'), dpi=150)
            plt.close()

        # 2. 靜態組的 Balanced Accuracy 曲線
        lc_models = [m for m in models if getattr(all_results[m], f'{target_name}_bal_lc_history') is not None]
        if lc_models:
            fig, axes = plt.subplots(1, len(lc_models), figsize=(6 * len(lc_models), 5))
            if len(lc_models) == 1: axes = [axes]
            
            fig.suptitle(f'第 {layer_idx+1} 層 - {target_name.upper()} 任務資料量 Balanced Accuracy 學習曲線', fontsize=16, fontweight='bold')
            
            for ax, model_name in zip(axes, lc_models):
                history = getattr(all_results[model_name], f'{target_name}_bal_lc_history')
                sizes = history['sizes']
                
                ax.plot(sizes, history['train'], 'o-', label='訓練集 (Train)', color='green', linewidth=2)
                ax.plot(sizes, history['val'], 'o-', label='交叉驗證集 (CV)', color='red', linewidth=2)
                
                ax.set_title(f'{model_name} 資料量與極限關係 (Balanced Acc)', fontsize=14)
                ax.set_xlabel('訓練樣本數 (Data Size)', fontsize=12)
                ax.set_ylabel('平衡準確率 (Balanced Accuracy)', fontsize=12)
                ax.legend(fontsize=10)
                ax.grid(True, linestyle='--', alpha=0.6)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'learning_curves_data_bal_acc_{target_name}.png'), dpi=150)
            plt.close()

    @staticmethod
    def plot_model_comparison(all_results, output_dir, target_name, layer_idx):
        """繪製各模型在測試集上的性能對比條形圖 (6 個指標)"""
        PlotGenerator.setup_chinese_font()
        models = list(all_results.keys())
        
        # 提取指標
        metrics_names = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'auc']
        display_names = {
            'accuracy': '準確率 (Accuracy)',
            'balanced_accuracy': '平衡準確率 (Bal Acc)',
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
        fig, axes = plt.subplots(1, 6, figsize=(30, 5))
        fig.suptitle(f'第 {layer_idx+1} 層 - {target_name.upper()} 任務模型性能對比', fontsize=16, fontweight='bold')
        
        colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974', '#8C564B']
        
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
    DATA_PATH = "experiment_results_train.pkl"
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
    y1, y2, y3 = preprocessor.create_targets()

    num_layers = X_3d.shape[1]
    
    for layer_idx in range(num_layers):
        print(f"\n" + "="*60)
        print(f"[開始訓練第 {layer_idx + 1} / {num_layers} 層特徵]")
        print("="*60)
        
        X_2d = X_3d[:, layer_idx, :]

        # 資料分割
        X_train_y1, X_val_y1, X_test_y1, y_train_y1, y_val_y1, y_test_y1, _ = DataSplitter.split_and_scale(X_2d, y1, layer_idx)
        X_train_y2, X_val_y2, X_test_y2, y_train_y2, y_val_y2, y_test_y2, _ = DataSplitter.split_and_scale(X_2d, y2, layer_idx)
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
            clf_y1_best, clf_y1_last, y1_pred, y1_proba, epoch_hist_y1, lc_hist_y1, bal_lc_hist_y1 = train_func(
                X_train_y1, X_val_y1, X_test_y1, y_train_y1, y_val_y1, y_test_y1, 'Y1'
            )
            results.y1_epoch_history = epoch_hist_y1
            results.y1_lc_history = lc_hist_y1
            results.y1_bal_lc_history = bal_lc_hist_y1
            results.y1_metrics = MetricsCalculator.calculate_metrics(y_test_y1, y1_pred, y1_proba)
            results.y1_test_true = y_test_y1
            results.y1_test_proba = y1_proba
            
            # 保存 y1 模型：分別保存 best 和 last
            joblib.dump(clf_y1_best, os.path.join(layer_output_dir, f"{model_name.lower()}_y1_best.pkl"))
            joblib.dump(clf_y1_last, os.path.join(layer_output_dir, f"{model_name.lower()}_y1_last.pkl"))
            
            # 訓練 y2
            clf_y2_best, clf_y2_last, y2_pred, y2_proba, epoch_hist_y2, lc_hist_y2, bal_lc_hist_y2 = train_func(
                X_train_y2, X_val_y2, X_test_y2, y_train_y2, y_val_y2, y_test_y2, 'Y2'
            )
            results.y2_epoch_history = epoch_hist_y2
            results.y2_lc_history = lc_hist_y2
            results.y2_bal_lc_history = bal_lc_hist_y2
            results.y2_metrics = MetricsCalculator.calculate_metrics(y_test_y2, y2_pred, y2_proba)
            results.y2_test_true = y_test_y2
            results.y2_test_proba = y2_proba
            
            # 保存 y2 模型：分別保存 best 和 last
            joblib.dump(clf_y2_best, os.path.join(layer_output_dir, f"{model_name.lower()}_y2_best.pkl"))
            joblib.dump(clf_y2_last, os.path.join(layer_output_dir, f"{model_name.lower()}_y2_last.pkl"))
            
            # 訓練 y3
            clf_y3_best, clf_y3_last, y3_pred, y3_proba, epoch_hist_y3, lc_hist_y3, bal_lc_hist_y3 = train_func(
                X_train_y3, X_val_y3, X_test_y3, y_train_y3, y_val_y3, y_test_y3, 'Y3'
            )
            results.y3_epoch_history = epoch_hist_y3
            results.y3_lc_history = lc_hist_y3
            results.y3_bal_lc_history = bal_lc_hist_y3
            results.y3_metrics = MetricsCalculator.calculate_metrics(y_test_y3, y3_pred, y3_proba)
            results.y3_test_true = y_test_y3
            results.y3_test_proba = y3_proba
            
            # 保存 y3 模型：分別保存 best 和 last
            joblib.dump(clf_y3_best, os.path.join(layer_output_dir, f"{model_name.lower()}_y3_best.pkl"))
            joblib.dump(clf_y3_last, os.path.join(layer_output_dir, f"{model_name.lower()}_y3_last.pkl"))
            
            all_results[model_name] = results

        # 輸出指標表格
        print(f"\n📊 [第 {layer_idx+1} 層] Y1 (模型回覆安全性任務) 性能指標:")
        metrics_df_y1 = pd.DataFrame({m: all_results[m].y1_metrics for m in all_results}).T
        print(metrics_df_y1.to_string())

        print(f"\n📊 [第 {layer_idx+1} 層] Y2 (提示詞有害性任務) 性能指標:")
        metrics_df_y2 = pd.DataFrame({m: all_results[m].y2_metrics for m in all_results}).T
        print(metrics_df_y2.to_string())

        print(f"\n📊 [第 {layer_idx+1} 層] Y3 (一致性任務) 性能指標:")
        metrics_df_y3 = pd.DataFrame({m: all_results[m].y3_metrics for m in all_results}).T
        print(metrics_df_y3.to_string())

        # 1. 先定義好三個子資料夾的路徑
        y1_img_dir = os.path.join(layer_output_dir, "y1_png")
        y2_img_dir = os.path.join(layer_output_dir, "y2_png")
        y3_img_dir = os.path.join(layer_output_dir, "y3_png")
        
        # 2. 自動建立資料夾 (exist_ok=True 會自動處理已存在的情況，不報錯)
        os.makedirs(y1_img_dir, exist_ok=True)
        os.makedirs(y2_img_dir, exist_ok=True)
        os.makedirs(y3_img_dir, exist_ok=True)
        
        # 3. 生成進階曲線與對比圖表 (直接傳入變數，程式碼超乾淨)
        print(f"\n[生成圖表] 正在繪製第 {layer_idx+1} 層的學習曲線與模型對比圖...")
        
        # Y1 的圖表
        PlotGenerator.plot_all_curves(all_results, y1_img_dir, 'y1', layer_idx)
        PlotGenerator.plot_model_comparison(all_results, y1_img_dir, 'y1', layer_idx)
        PlotGenerator.plot_roc_curve(all_results, y1_img_dir, 'y1', layer_idx)
        PlotGenerator.plot_balanced_accuracy_curves(all_results, y1_img_dir, 'y1', layer_idx)
        
        # Y2 的圖表
        PlotGenerator.plot_all_curves(all_results, y2_img_dir, 'y2', layer_idx)
        PlotGenerator.plot_model_comparison(all_results, y2_img_dir, 'y2', layer_idx)
        PlotGenerator.plot_roc_curve(all_results, y2_img_dir, 'y2', layer_idx)
        PlotGenerator.plot_balanced_accuracy_curves(all_results, y2_img_dir, 'y2', layer_idx)
        
        # Y3 的圖表
        PlotGenerator.plot_all_curves(all_results, y3_img_dir, 'y3', layer_idx)
        PlotGenerator.plot_model_comparison(all_results, y3_img_dir, 'y3', layer_idx)
        PlotGenerator.plot_roc_curve(all_results, y3_img_dir, 'y3', layer_idx)
        PlotGenerator.plot_balanced_accuracy_curves(all_results, y3_img_dir, 'y3', layer_idx)
        
    print(f"\n[OK] 所有 {num_layers} 層特徵的模型訓練、評估與繪圖已全部完成！")
    print(f"結果與圖表已儲存至 {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
