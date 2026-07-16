import os
import sys
import numpy as np
import joblib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Import from project modules
from unified_train import DataPreprocessor, DataSplitter
import utils_calibration
from wrapper_models import CorrectnessClassifierWrapper

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

def generate_grid_plot(df, dataset_name, title, save_path):
    """
    Plots a 3x3 grid comparing SGD, MLP, LGB, LR, RF across layers.
    Columns: Brier Score, ECE, Log Loss
    Rows: y1, y2, y3
    """
    utils_calibration.setup_chinese_font()
    models = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
    colors = {'SGD': '#4C72B0', 'MLP': '#55A868', 'LGB': '#C44E52', 'LR': '#8172B3', 'RF': '#CCB974'}
    markers = {'SGD': 'o', 'MLP': 's', 'LGB': '^', 'LR': 'v', 'RF': 'D'}
    
    targets = ['y1', 'y2', 'y3']
    target_names = {
        'y1': 'y1 (模型回覆安全性)',
        'y2': 'y2 (提示詞有害性)',
        'y3': 'y3 (安全判定一致性)'
    }
    metrics = ['Brier Score', 'ECE', 'Log Loss']
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15), sharex=True)
    fig.suptitle(f'{title} - {dataset_name}', fontsize=20, fontweight='bold', y=0.96)
    
    for r_idx, tg in enumerate(targets):
        for c_idx, mt in enumerate(metrics):
            ax = axes[r_idx, c_idx]
            
            df_sub = df[(df['target'] == tg) & (df['metric'] == mt)]
            
            best_val = float('inf')
            best_model = None
            best_layer = None
            
            for model in models:
                df_model = df_sub[df_sub['model'] == model].sort_values('layer')
                if df_model.empty:
                    continue
                
                layers = df_model['layer'].values
                vals = df_model['value'].values
                
                ax.plot(layers, vals, label=model, color=colors[model], 
                        marker=markers[model], markersize=6, linewidth=1.5, alpha=0.9)
                
                # Track best
                min_idx = np.argmin(vals)
                if vals[min_idx] < best_val:
                    best_val = vals[min_idx]
                    best_model = model
                    best_layer = layers[min_idx]
            
            # Highlight best model with a golden star
            if best_model is not None:
                ax.plot(best_layer, best_val, marker='*', color='#FFD700', 
                        markersize=14, markeredgecolor='black', markeredgewidth=1, 
                        zorder=5, label='最佳模型')
            
            if r_idx == 0:
                ax.set_title(f'{mt}', fontsize=14, fontweight='bold')
            if c_idx == 0:
                ax.set_ylabel(f'{target_names[tg]}\n指標值 (越低越好)', fontsize=12, fontweight='bold')
            if r_idx == 2:
                ax.set_xlabel('特徵層數 (Layer)', fontsize=12)
                
            num_layers = int(df['layer'].max()) if not df.empty else 6
            ax.set_xticks(range(1, num_layers + 1))
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # Remove duplicate labels in legend
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=9)
            
    plt.tight_layout()
    fig.subplots_adjust(top=0.90)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def main():
    base_output_dir = "results/correctness_metrics_comparison"
    os.makedirs(base_output_dir, exist_ok=True)
    sys.stdout = DualLogger(os.path.join(base_output_dir, "metrics_log.txt"))
    
    print("="*80)
    print(" 啟動 Step 3: 多維度指標對比繪圖管線 (10,000 數據模型全覆蓋版)")
    print("="*80)
    
    # 1. Load original datasets to reconstruct splits
    TRAIN_PATH = "experiment_results_train_10000.pkl"
    EVAL_PATH = "experiment_results_eval.pkl"
    
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(EVAL_PATH):
        print(f"錯誤: 確保 {TRAIN_PATH} 與 {EVAL_PATH} 存在。")
        sys.exit(1)
        
    print(f"[1] 載入基準 10000 訓練集與 Eval 集...")
    prep_train = DataPreprocessor(TRAIN_PATH)
    df_10000 = prep_train.load_data()
    X_3d_train = prep_train.extract_features()
    y_targets_train = prep_train.create_targets()
    # Pre-calculate target labels for df_10000 using preprocessor results to avoid consistency issues
    df_10000['y1'] = y_targets_train[0]
    df_10000['y2'] = y_targets_train[1]
    df_10000['y3'] = y_targets_train[2]
    
    prep_eval = DataPreprocessor(EVAL_PATH)
    prep_eval.load_data()
    X_3d_eval = prep_eval.extract_features()
    y_targets_eval = prep_eval.create_targets()
    
    # 2. Load aligned and augmented datasets generated by run_02
    ALIGNED_TEST1_PATH = "aligned_test1.pkl"
    ALIGNED_TEST2_PATH = "aligned_test2.pkl"
    AUGMENTED_TEST1_PATH = "augmented_test1.pkl"
    AUGMENTED_TEST2_PATH = "augmented_test2.pkl"
    
    if not (os.path.exists(ALIGNED_TEST1_PATH) and os.path.exists(ALIGNED_TEST2_PATH) and
            os.path.exists(AUGMENTED_TEST1_PATH) and os.path.exists(AUGMENTED_TEST2_PATH)):
        print(f"錯誤: 找不到擴增或對齊後的資料集檔案。請先執行 run_02_data_augmentation_experiments.py。")
        sys.exit(1)
        
    print(f"[2] 載入對齊與擴增資料集...")
    aligned_test1_dict = joblib.load(ALIGNED_TEST1_PATH)
    aligned_test2_dict = joblib.load(ALIGNED_TEST2_PATH)
    augmented_test1_dict = joblib.load(AUGMENTED_TEST1_PATH)
    augmented_test2_dict = joblib.load(AUGMENTED_TEST2_PATH)
    
    num_layers = X_3d_train.shape[1]
    models = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
    targets = ['y1', 'y2', 'y3']
    target_names = {
        'y1': 'y1 (模型回覆安全性預測)',
        'y2': 'y2 (提示詞有害性預測)',
        'y3': 'y3 (安全判定一致性預測)'
    }
    
    # Data lists to build dataframes for plotting
    std_model_records = []
    aug_model_records = []
    align_model_records = []
    
    # Target loops
    for target_idx, target_name in enumerate(targets):
        y_train = y_targets_train[target_idx]
        y_eval = y_targets_eval[target_idx]
        y_eval_np = np.array(y_eval)
        
        # Original splits index
        train_val_idx, test_idx = train_test_split(
            df_10000.index, test_size=0.2, random_state=42, stratify=y_train
        )
        test1_idx, test2_idx = train_test_split(
            test_idx, test_size=0.5, random_state=42
        )
        
        df_test1_orig = df_10000.loc[test1_idx]
        df_test2_orig = df_10000.loc[test2_idx]
        y_test1_orig = df_test1_orig[target_name].values
        y_test2_orig = df_test2_orig[target_name].values
        
        # Load augmented datasets
        df_test1_aug = augmented_test1_dict[target_name]
        df_test2_aug = augmented_test2_dict[target_name]
        y_test1_aug = df_test1_aug[target_name].values
        y_test2_aug = df_test2_aug[target_name].values
        
        # Load aligned datasets
        df_test1_align = aligned_test1_dict[target_name]
        df_test2_align = aligned_test2_dict[target_name]
        y_test1_align = df_test1_align[target_name].values
        y_test2_align = df_test2_align[target_name].values
        
        print(f"\n計算目標變數 {target_name} 的各模型評估指標...")
        
        for layer_idx in range(num_layers):
            layer_num = layer_idx + 1
            
            # Extract layer features
            X_eval_layer = X_3d_eval[:, layer_idx, :]
            
            X_test1_orig_layer = np.array(df_test1_orig['hidden_state'].tolist())[:, layer_idx, :]
            X_test2_orig_layer = np.array(df_test2_orig['hidden_state'].tolist())[:, layer_idx, :]
            
            X_test1_aug_layer = np.array(df_test1_aug['hidden_state'].tolist())[:, layer_idx, :]
            X_test2_aug_layer = np.array(df_test2_aug['hidden_state'].tolist())[:, layer_idx, :]
            
            X_test1_align_layer = np.array(df_test1_align['hidden_state'].tolist())[:, layer_idx, :]
            X_test2_align_layer = np.array(df_test2_align['hidden_state'].tolist())[:, layer_idx, :]
            
            for model_name in models:
                # Load baseline calibrated model (std)
                std_model_path = f"results/unified_training/layer_{layer_num}/{model_name.lower()}_{target_name.lower()}_calibrated.pkl"
                # Load augmented calibrated model (aug)
                aug_model_path = f"results/unified_training/layer_{layer_num}/{model_name.lower()}_{target_name.lower()}_calibrated_augmented.pkl"
                # Load aligned calibrated model (align)
                align_model_path = f"results/unified_training/layer_{layer_num}/{model_name.lower()}_{target_name.lower()}_calibrated_aligned.pkl"
                
                if not (os.path.exists(std_model_path) and os.path.exists(aug_model_path) and os.path.exists(align_model_path)):
                    continue
                    
                cal_std = joblib.load(std_model_path)
                cal_aug = joblib.load(aug_model_path)
                cal_align = joblib.load(align_model_path)
                
                # Load the original base model to generate correctness targets
                base_model_path = f"results/unified_training/layer_{layer_num}/{model_name.lower()}_{target_name.lower()}_best.pkl"
                base_clf = joblib.load(base_model_path)
                
                if target_name in ['y1', 'y2']:
                    pred_eval = base_clf.predict(X_eval_layer)
                    pred_test1_orig = base_clf.predict(X_test1_orig_layer)
                    pred_test2_orig = base_clf.predict(X_test2_orig_layer)
                    
                    pred_test1_aug = base_clf.predict(X_test1_aug_layer)
                    pred_test2_aug = base_clf.predict(X_test2_aug_layer)
                    
                    pred_test1_align = base_clf.predict(X_test1_align_layer)
                    pred_test2_align = base_clf.predict(X_test2_align_layer)
                    
                    y_eval_correct = (pred_eval == y_eval_np).astype(int)
                    y_test1_orig_correct = (pred_test1_orig == y_test1_orig).astype(int)
                    y_test2_orig_correct = (pred_test2_orig == y_test2_orig).astype(int)
                    
                    y_test1_aug_correct = (pred_test1_aug == y_test1_aug).astype(int)
                    y_test2_aug_correct = (pred_test2_aug == y_test2_aug).astype(int)
                    
                    y_test1_align_correct = (pred_test1_align == y_test1_align).astype(int)
                    y_test2_align_correct = (pred_test2_align == y_test2_align).astype(int)
                else:
                    y_eval_correct = y_eval_np
                    y_test1_orig_correct = y_test1_orig
                    y_test2_orig_correct = y_test2_orig
                    
                    y_test1_aug_correct = y_test1_aug
                    y_test2_aug_correct = y_test2_aug
                    
                    y_test1_align_correct = y_test1_align
                    y_test2_align_correct = y_test2_align
                
                # ==================== Evaluate Old Calibrated Model (model_std) ====================
                datasets_std = {
                    'data_std_test1': (X_test1_orig_layer, y_test1_orig_correct),
                    'data_std_test2': (X_test2_orig_layer, y_test2_orig_correct),
                    'data_eval': (X_eval_layer, y_eval_correct)
                }
                for ds_key, (X, y) in datasets_std.items():
                    probs = cal_std.predict_proba(X)[:, 1]
                    metrics = utils_calibration.calculate_all_metrics(y, probs)
                    
                    for m_name, m_val in [('Brier Score', metrics['brier']), 
                                          ('ECE', metrics['ece']), 
                                          ('Log Loss', metrics['logloss'])]:
                        std_model_records.append({
                            'target': target_name,
                            'layer': layer_num,
                            'model': model_name,
                            'dataset': ds_key,
                            'metric': m_name,
                            'value': m_val
                        })
                
                # ==================== Evaluate Augmented Calibrated Model (model_aug) ====================
                datasets_aug = {
                    'data_aug_test1': (X_test1_aug_layer, y_test1_aug_correct),
                    'data_aug_test2': (X_test2_aug_layer, y_test2_aug_correct),
                    'data_eval': (X_eval_layer, y_eval_correct)
                }
                for ds_key, (X, y) in datasets_aug.items():
                    probs = cal_aug.predict_proba(X)[:, 1]
                    metrics = utils_calibration.calculate_all_metrics(y, probs)
                    
                    for m_name, m_val in [('Brier Score', metrics['brier']), 
                                          ('ECE', metrics['ece']), 
                                          ('Log Loss', metrics['logloss'])]:
                        aug_model_records.append({
                            'target': target_name,
                            'layer': layer_num,
                            'model': model_name,
                            'dataset': ds_key,
                            'metric': m_name,
                            'value': m_val
                        })
                
                # ==================== Evaluate Aligned Calibrated Model (model_align) ====================
                datasets_align = {
                    'data_align_test1': (X_test1_align_layer, y_test1_align_correct),
                    'data_std_test2': (X_test2_orig_layer, y_test2_orig_correct), # original test2 (Requirement 9)
                    'data_align_test2': (X_test2_align_layer, y_test2_align_correct),
                    'data_eval': (X_eval_layer, y_eval_correct)
                }
                for ds_key, (X, y) in datasets_align.items():
                    probs = cal_align.predict_proba(X)[:, 1]
                    metrics = utils_calibration.calculate_all_metrics(y, probs)
                    
                    for m_name, m_val in [('Brier Score', metrics['brier']), 
                                          ('ECE', metrics['ece']), 
                                          ('Log Loss', metrics['logloss'])]:
                        align_model_records.append({
                            'target': target_name,
                            'layer': layer_num,
                            'model': model_name,
                            'dataset': ds_key,
                            'metric': m_name,
                            'value': m_val
                        })
                        
            print(f"  └─ 第 {layer_num} 層計算完成")
            
    df_std = pd.DataFrame(std_model_records)
    df_aug = pd.DataFrame(aug_model_records)
    df_align = pd.DataFrame(align_model_records)
    
    # 3. Generate line comparison grids (Requirement 9)
    print("\n[繪圖] 正在生成標準模型 (model_std) 的 3x3 指標對比圖...")
    for ds in ['data_std_test1', 'data_std_test2', 'data_eval']:
        df_ds = df_std[df_std['dataset'] == ds]
        save_path = os.path.join(base_output_dir, "std_model_comparison", f"calibration_comparison_{ds}.png")
        generate_grid_plot(df_ds, ds, "標準校正模型指標對比圖 (model_std)", save_path)
        print(f"  └─ 儲存至: {save_path}")
        
    print("\n[繪圖] 正在生成增量模型 (model_aug) 的 3x3 指標對比圖...")
    for ds in ['data_aug_test1', 'data_aug_test2', 'data_eval']:
        df_ds = df_aug[df_aug['dataset'] == ds]
        save_path = os.path.join(base_output_dir, "aug_model_comparison", f"calibration_comparison_{ds}.png")
        generate_grid_plot(df_ds, ds, "增量校正模型指標對比圖 (model_aug)", save_path)
        print(f"  └─ 儲存至: {save_path}")
        
    print("\n[繪圖] 正在生成對齊模型 (model_align) 的 3x3 指標對比圖...")
    for ds in ['data_align_test1', 'data_std_test2', 'data_align_test2', 'data_eval']:
        df_ds = df_align[df_align['dataset'] == ds]
        save_path = os.path.join(base_output_dir, "align_model_comparison", f"calibration_comparison_{ds}.png")
        generate_grid_plot(df_ds, ds, "對齊校正模型指標對比圖 (model_align)", save_path)
        print(f"  └─ 儲存至: {save_path}")
        
    # Print out summary of findings in log file
    print("\n" + "="*80)
    print(" 數據校正對比總結報表")
    print("="*80)
    
    for mt in ['ECE', 'Brier Score']:
        print(f"\n[{mt} 最佳模型排名 (在外部驗證集 data_eval 上)]")
        # std model
        df_std_eval = df_std[(df_std['dataset'] == 'data_eval') & (df_std['metric'] == mt)]
        if not df_std_eval.empty:
            best_std = df_std_eval.loc[df_std_eval['value'].idxmin()]
            print(f"  ▶ 標準校正模型組: {best_std['model']} (層數: {best_std['layer']}, 任務: {best_std['target']}) - 數值: {best_std['value']:.4f}")
        
        # aug model
        df_aug_eval = df_aug[(df_aug['dataset'] == 'data_eval') & (df_aug['metric'] == mt)]
        if not df_aug_eval.empty:
            best_aug = df_aug_eval.loc[df_aug_eval['value'].idxmin()]
            print(f"  ▶ 增量校正模型組: {best_aug['model']} (層數: {best_aug['layer']}, 任務: {best_aug['target']}) - 數值: {best_aug['value']:.4f}")
            
        # align model
        df_align_eval = df_align[(df_align['dataset'] == 'data_eval') & (df_align['metric'] == mt)]
        if not df_align_eval.empty:
            best_align = df_align_eval.loc[df_align_eval['value'].idxmin()]
            print(f"  ▶ 對齊校正模型組: {best_align['model']} (層數: {best_align['layer']}, 任務: {best_align['target']}) - 數值: {best_align['value']:.4f}")
        
    print("\n" + "="*80)
    print(" Step 3: 運行完成！所有對比大圖已儲存。")
    print("="*80)

if __name__ == "__main__":
    main()
