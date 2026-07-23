"""
統一機器學習模型訓練框架 (v2_20k 全量版)
========================================
功能：針對 v2_20k 資料集，在 Layer 1~6 完整訓練 5 種分類模型 (SGD, MLP, LGB, LR, RF)
產出：
  1. 最佳模型檔至 models/v2_20k/layer_1~6/
  2. 第一階段訓練視覺化圖檔至 results/v2_20k/01_Unified_Training/
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, log_loss, roc_curve, auc, balanced_accuracy_score
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

warnings.filterwarnings('ignore')

def setup_chinese_font():
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'PMingLiU', 'DFKai-SB', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

class DualLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")
    def write(self, message):
        try:
            self.terminal.write(message)
        except UnicodeEncodeError:
            encoding = getattr(self.terminal, 'encoding', 'utf-8') or 'utf-8'
            self.terminal.write(message.encode(encoding, errors='replace').decode(encoding))
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

class ModelResults:
    def __init__(self, model_name):
        self.model_name = model_name
        self.y1_metrics = {}
        self.y2_metrics = {}
        self.y3_metrics = {}
        self.y1_epoch_history = None
        self.y2_epoch_history = None
        self.y3_epoch_history = None
        self.y1_test_true = None
        self.y1_test_proba = None
        self.y2_test_true = None
        self.y2_test_proba = None
        self.y3_test_true = None
        self.y3_test_proba = None

class UnifiedModelTrainerV2:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def train_sgd(self, X_train, X_val, X_test, y_train, y_val, y_test, y_name):
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
            loss='log_loss', penalty='l2', alpha=0.01,
            learning_rate='adaptive', eta0=0.0001,
            class_weight=class_weight_dict, random_state=42
        )
        
        epochs = 100
        batch_size = 64
        history = {'sizes': [], 'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
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
            train_proba = clf.predict_proba(X_train_pca)
            val_proba = clf.predict_proba(X_val_pca)
            train_loss = log_loss(y_train_res, train_proba, labels=classes)
            val_loss = log_loss(y_val, val_proba, labels=classes)

            history['sizes'].append(epoch + 1)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_clf = copy.deepcopy(clf)
                best_epoch = epoch + 1
                
        if best_clf is None: best_clf = clf
        history['best_epoch'] = best_epoch

        y_pred = best_clf.predict(X_test_pca)
        y_pred_proba = best_clf.predict_proba(X_test_pca)[:, 1]
        pipeline = ImbPipeline([('scaler', scaler), ('pca', pca), ('clf', best_clf)])
        return pipeline, y_pred, y_pred_proba, history

    def train_mlp(self, X_train, X_val, X_test, y_train, y_val, y_test, y_name):
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
        history = {'sizes': [], 'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
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
            train_proba = clf.predict_proba(X_train_pca)
            val_proba = clf.predict_proba(X_val_pca)
            train_loss = log_loss(y_train_res, train_proba, labels=classes)
            val_loss = log_loss(y_val, val_proba, labels=classes)

            history['sizes'].append(epoch + 1)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_clf = copy.deepcopy(clf)
                best_epoch = epoch + 1
                
        if best_clf is None: best_clf = clf
        history['best_epoch'] = best_epoch

        y_pred = best_clf.predict(X_test_pca)
        y_pred_proba = best_clf.predict_proba(X_test_pca)[:, 1]
        pipeline = ImbPipeline([('scaler', scaler), ('pca', pca), ('clf', best_clf)])
        return pipeline, y_pred, y_pred_proba, history

    def train_lgb(self, X_train, X_val, X_test, y_train, y_val, y_test, y_name):
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
            n_estimators=100, learning_rate=0.05, random_state=42,
            max_depth=10, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05, verbose=-1
        )
        clf.fit(
            X_train_pca, y_train_res,
            eval_set=[(X_val_pca, y_val)],
            eval_names=['val'],
            eval_metric='binary_logloss',
            callbacks=[lgb.log_evaluation(period=0)]
        )
        val_losses = clf.evals_result_['val']['binary_logloss']
        best_iteration = int(np.argmin(val_losses) + 1)
        best_clf = copy.deepcopy(clf)
        best_clf._best_iteration = best_iteration

        y_pred = best_clf.predict(X_test_pca)
        y_pred_proba = best_clf.predict_proba(X_test_pca)[:, 1]
        pipeline = ImbPipeline([('scaler', scaler), ('pca', pca), ('clf', best_clf)])
        history = {'best_epoch': best_iteration}
        return pipeline, y_pred, y_pred_proba, history

    def train_lr(self, X_train, X_val, X_test, y_train, y_val, y_test, y_name):
        pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('sampler', RandomUnderSampler(random_state=42)),
            ('pca', PCA(n_components=128, random_state=42)),
            ('clf', LogisticRegression(C=0.01, penalty='l2', max_iter=1000, random_state=42))
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        return pipeline, y_pred, y_pred_proba, None

    def train_rf(self, X_train, X_val, X_test, y_train, y_val, y_test, y_name):
        pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('sampler', RandomUnderSampler(random_state=42)),
            ('pca', PCA(n_components=128, random_state=42)),
            ('clf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=2))
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        return pipeline, y_pred, y_pred_proba, None

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
    except Exception:
        metrics['auc'] = np.nan
    return metrics

def plot_stage1_figures(all_results, output_dir, target_name, layer_num):
    setup_chinese_font()
    os.makedirs(output_dir, exist_ok=True)
    models = list(all_results.keys())
    
    # 1. ROC Curves
    plt.figure(figsize=(8, 7))
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5, label='隨機猜測 (AUC = 0.500)')
    colors = {'SGD': '#1f77b4', 'MLP': '#ff7f0e', 'LGB': '#2ca02c', 'LR': '#d62728', 'RF': '#9467bd'}
    
    for model_name in models:
        res = all_results[model_name]
        y_true = getattr(res, f'{target_name}_test_true')
        y_proba = getattr(res, f'{target_name}_test_proba')
        if y_true is not None and y_proba is not None:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            auc_score = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors.get(model_name, '#333333'), linewidth=2, label=f'{model_name} (AUC = {auc_score:.3f})')
            
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('偽陽性率 (False Positive Rate)', fontsize=12)
    plt.ylabel('真陽性率 (True Positive Rate)', fontsize=12)
    plt.title(f'第 {layer_num} 層 - {target_name.upper()} 任務模型 ROC 曲線對比', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'roc_curve_layer_{layer_num}_{target_name}.png'), dpi=150)
    plt.close()
    
    # 2. Model Comparison 6 Bars
    metrics_names = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'auc']
    display_names = {'accuracy': '準確率', 'balanced_accuracy': '平衡準確率', 'precision': '精確率', 'recall': '召回率', 'f1': 'F1 值', 'auc': 'AUC 值'}
    
    fig, axes = plt.subplots(1, 6, figsize=(30, 5))
    fig.suptitle(f'第 {layer_num} 層 - {target_name.upper()} 任務模型性能對比', fontsize=16, fontweight='bold')
    bar_colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974']
    
    for idx, metric_name in enumerate(metrics_names):
        ax = axes[idx]
        scores = [getattr(all_results[m], f'{target_name}_metrics').get(metric_name, np.nan) for m in models]
        bars = ax.bar(models, scores, color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.7)
        ax.set_title(display_names[metric_name], fontsize=13, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'model_comparison_layer_{layer_num}_{target_name}.png'), dpi=150)
    plt.close()

def main():
    data_dir = "data/v2_20k"
    models_dir = "models/v2_20k"
    results_base = "results/v2_20k/01_Unified_Training"
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_base, exist_ok=True)
    
    sys.stdout = DualLogger(os.path.join(results_base, "training_log.txt"))
    
    print("="*80)
    print("v2_20k 全量全層模型訓練 (Layer 1~6, Models: SGD/MLP/LGB/LR/RF)")
    print("="*80)
    
    train_path = os.path.join(data_dir, "train_12000.pkl")
    val_path = os.path.join(data_dir, "val_4000.pkl")
    test1_path = os.path.join(data_dir, "test1_2000.pkl")
    
    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test1_path)):
        print("錯誤: 找不到 data/v2_20k/ 資料檔，請先執行 prepare_v2_20k_data.py")
        sys.exit(1)
        
    df_train = pd.read_pickle(train_path)
    df_val = pd.read_pickle(val_path)
    df_test1 = pd.read_pickle(test1_path)
    
    X_3d_train = np.array(df_train['hidden_state'].tolist())
    X_3d_val = np.array(df_val['hidden_state'].tolist())
    X_3d_test = np.array(df_test1['hidden_state'].tolist())
    
    targets_map = {
        'y1': (df_train['y1'].values, df_val['y1'].values, df_test1['y1'].values),
        'y2': (df_train['y2'].values, df_val['y2'].values, df_test1['y2'].values),
        'y3': (df_train['y3'].values, df_val['y3'].values, df_test1['y3'].values)
    }
    
    models_list = [('SGD', 'train_sgd'), ('MLP', 'train_mlp'), ('LGB', 'train_lgb'), ('LR', 'train_lr'), ('RF', 'train_rf')]
    
    num_layers = X_3d_train.shape[1]
    
    for layer_num in range(1, num_layers + 1):
        print(f"\n============================================================")
        print(f"[開始訓練第 {layer_num} / {num_layers} 層特徵 (Layer {layer_num})]")
        print(f"============================================================")
        
        layer_output_dir = os.path.join(models_dir, f"layer_{layer_num}")
        os.makedirs(layer_output_dir, exist_ok=True)
        trainer = UnifiedModelTrainerV2(output_dir=layer_output_dir)
        
        X_tr = X_3d_train[:, layer_num - 1, :]
        X_va = X_3d_val[:, layer_num - 1, :]
        X_te = X_3d_test[:, layer_num - 1, :]
        
        all_results = {}
        
        for model_name, train_fn_name in models_list:
            train_fn = getattr(trainer, train_fn_name)
            results = ModelResults(model_name)
            
            for target_name in ['y1', 'y2', 'y3']:
                y_tr, y_va, y_te = targets_map[target_name]
                print(f"  └─ [Layer {layer_num}] 訓練 {model_name} ({target_name.upper()})...")
                
                pipeline, y_pred, y_proba, history = train_fn(X_tr, X_va, X_te, y_tr, y_va, y_te, target_name)
                
                # Save best model
                save_file = os.path.join(layer_output_dir, f"{model_name.lower()}_{target_name}_best.pkl")
                joblib.dump(pipeline, save_file)
                
                # Record metrics & probabilities
                setattr(results, f'{target_name}_metrics', calculate_metrics(y_te, y_pred, y_proba))
                setattr(results, f'{target_name}_test_true', y_te)
                setattr(results, f'{target_name}_test_proba', y_proba)
                setattr(results, f'{target_name}_epoch_history', history)
                
            all_results[model_name] = results
            
        # 產出 Stage 1 視覺化圖表
        print(f"\n[繪製 Layer {layer_num} 階段一圖表...]")
        roc_dir = os.path.join(results_base, "ROC_Curves")
        comp_dir = os.path.join(results_base, "Model_Comparison")
        for target_name in ['y1', 'y2', 'y3']:
            plot_stage1_figures(all_results, roc_dir, target_name, layer_num)
            plot_stage1_figures(all_results, comp_dir, target_name, layer_num)
            
    print("\n所有 6 層全量模型訓練與階段一圖檔繪製完成！")

if __name__ == '__main__':
    main()
