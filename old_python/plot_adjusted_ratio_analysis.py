import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import gc
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, log_loss

# Ensure stdout and stderr use UTF-8 on Windows to prevent CP950 encoding errors
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add parent path or use local path imports
sys.path.append(os.path.abspath("."))
from unified_train import DataPreprocessor, DataSplitter

class DualLogger:
    """Writes output both to stdout and to a log file."""
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

    def close(self):
        self.log.close()

def setup_chinese_font():
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'PMingLiU', 'DFKai-SB', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False 

def calculate_ece(y_true, y_prob, n_bins=10):
    """Calculate Expected Calibration Error (ECE)."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(y_prob, bin_edges) - 1
    
    ece = 0.0
    n_total = len(y_prob)
    
    for i in range(n_bins):
        bin_mask = (binids == i)
        if i == n_bins - 1:
            bin_mask = bin_mask | (y_prob == 1.0)
            
        n_bin = np.sum(bin_mask)
        if n_bin > 0:
            acc = np.mean(y_true[bin_mask])
            conf = np.mean(y_prob[bin_mask])
            ece += (n_bin / n_total) * np.abs(acc - conf)
            
    return ece

def supplement_to_ratio(X_test, y_test, df_pool, pool_idx_1, pool_idx_0, 
                        target_ratio_positive, target_col, layer_idx, random_state=42):
    """
    Supplements X_test, y_test with non-duplicate samples from df_pool
    so that the final positive class ratio matches target_ratio_positive.
    Efficiently converts only selected rows to numpy representation.
    """
    idx_1_test = np.where(y_test == 1)[0]
    idx_0_test = np.where(y_test == 0)[0]
    
    n_1 = len(idx_1_test)
    n_0 = len(idx_0_test)
    
    p_current = n_1 / (n_1 + n_0) if (n_1 + n_0) > 0 else 0
    rng = np.random.default_rng(random_state)
    
    selected_pool_indices = []
    
    if target_ratio_positive > p_current:
        # Supplement positive class (class 1)
        # Target formula: (n_1 + delta_1) / (n_1 + delta_1 + n_0) = target_ratio_positive
        target_n_1 = int(round(n_0 * target_ratio_positive / (1.0 - target_ratio_positive)))
        delta_1 = target_n_1 - n_1
        
        if delta_1 > 0:
            if delta_1 > len(pool_idx_1):
                print(f"  [警告] 資源池中正樣本不足！需要: {delta_1}, 可用: {len(pool_idx_1)}")
                delta_1 = len(pool_idx_1)
            
            selected_sub_idx = rng.choice(len(pool_idx_1), size=delta_1, replace=False)
            selected_pool_indices = [pool_idx_1[i] for i in selected_sub_idx]
            
            # Remove selected indices from the pool
            for i in sorted(selected_sub_idx, reverse=True):
                del pool_idx_1[i]
    else:
        # Supplement negative class (class 0)
        # Target formula: n_1 / (n_1 + n_0 + delta_0) = target_ratio_positive
        target_n_0 = int(round(n_1 * (1.0 - target_ratio_positive) / target_ratio_positive))
        delta_0 = target_n_0 - n_0
        
        if delta_0 > 0:
            if delta_0 > len(pool_idx_0):
                print(f"  [警告] 資源池中負樣本不足！需要: {delta_0}, 可用: {len(pool_idx_0)}")
                delta_0 = len(pool_idx_0)
            
            selected_sub_idx = rng.choice(len(pool_idx_0), size=delta_0, replace=False)
            selected_pool_indices = [pool_idx_0[i] for i in selected_sub_idx]
            
            # Remove selected indices from the pool
            for i in sorted(selected_sub_idx, reverse=True):
                del pool_idx_0[i]
                
    if len(selected_pool_indices) > 0:
        # Fetch the selected pool dataframe rows
        df_selected = df_pool.loc[selected_pool_indices]
        
        # Convert only selected hidden states to numpy
        X_pool_selected_3d = np.array(df_selected['hidden_state'].tolist())
        X_pool_selected_2d = X_pool_selected_3d[:, layer_idx, :]
        
        y_pool_selected = df_selected[target_col].values
        
        # Combine
        X_new = np.concatenate([X_test, X_pool_selected_2d], axis=0)
        y_new = np.concatenate([y_test, y_pool_selected], axis=0)
    else:
        X_new, y_new = X_test.copy(), y_test.copy()
        
    shuffle_idx = np.arange(len(y_new))
    rng.shuffle(shuffle_idx)
    return X_new[shuffle_idx], y_new[shuffle_idx]

def get_calibrated_classifier(clf, method='isotonic'):
    try:
        from sklearn.frozen import FrozenEstimator
        return CalibratedClassifierCV(FrozenEstimator(clf), method=method, cv=None)
    except ImportError:
        return CalibratedClassifierCV(clf, method=method, cv='prefit')

def generate_individual_reliability_plot(models_data, title, save_path):
    """Generates reliability curve diagrams comparing Raw vs Std vs Adj calibration."""
    setup_chinese_font()
    n_models = len(models_data)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]
        
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    colors = {'Raw': '#7F7F7F', 'StdCal': '#1F77B4', 'AdjCal': '#D62728'}
    markers = {'Raw': 'x', 'StdCal': 'o', 'AdjCal': 's'}
    
    for ax, (model_name, schemes) in zip(axes, models_data.items()):
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", alpha=0.7)
        
        for scheme_name, (y_true, y_prob) in schemes.items():
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
            ece = calculate_ece(y_true, y_prob)
            ax.plot(prob_pred, prob_true, marker=markers[scheme_name], color=colors[scheme_name], 
                    label=f"{scheme_name} (ECE: {ece:.4f})", linewidth=1.5)
            
        ax.set_title(model_name, fontsize=12, fontweight='bold')
        ax.set_xlabel("Mean predicted probability", fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel("Fraction of positives", fontsize=10)
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    # Setup directories
    output_dir = "results/adjusted_ratio_calibration"
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(output_dir, "calibration_metrics_log.txt")
    logger = DualLogger(log_file)
    sys.stdout = logger
    
    print("="*80)
    print(" 啟動資料補充校正實驗管線 (以額外不重複資料補足 Test1/Test2 比例以對齊 Eval)")
    print("="*80)
    
    # 1. Load datasets
    TRAIN_PATH = "experiment_results_train_10000.pkl"
    FULL_PATH = "experiment_results_train.pkl"
    EVAL_PATH = "experiment_results_eval.pkl"
    
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(FULL_PATH) or not os.path.exists(EVAL_PATH):
        print(f"錯誤: 請確保 {TRAIN_PATH}、{FULL_PATH} 與 {EVAL_PATH} 存在於當前工作目錄。")
        sys.exit(1)
        
    print(f"[1] 載入基準 10000 訓練集: {TRAIN_PATH}")
    df_10000 = pd.read_pickle(TRAIN_PATH)
    prep_train = DataPreprocessor(TRAIN_PATH)
    prep_train.load_data()
    X_3d_train = prep_train.extract_features()
    y_targets_train = prep_train.create_targets() # y1, y2, y3
    
    print(f"[2] 載入 85000 全量訓練集 (並篩選出未重複的資料池)...")
    df_full = pd.read_pickle(FULL_PATH)
    # Filter out overlap to get unused pool
    df_unused = df_full[~df_full['id'].isin(df_10000['id'])].copy()
    
    # Pre-add target columns to df_unused
    df_unused['y1'] = df_unused['model_reply'].str.lower().str.contains('unsafe').astype(int)
    df_unused['y2'] = df_unused['data_type'].str.contains('harmful').astype(int)
    df_unused['y3'] = (df_unused['y1'] == df_unused['y2']).astype(int)
    
    print(f"  └─ 全量訓練集: {len(df_full)} 筆，基準訓練集: {len(df_10000)} 筆，不重複可用資源池: {len(df_unused)} 筆")
    
    # Delete df_full immediately to free memory
    del df_full
    gc.collect()
    
    print(f"[3] 載入外部驗證集: {EVAL_PATH}")
    prep_eval = DataPreprocessor(EVAL_PATH)
    prep_eval.load_data()
    X_3d_eval = prep_eval.extract_features()
    y_targets_eval = prep_eval.create_targets()
    
    num_layers = X_3d_train.shape[1]
    models = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
    targets = ['y1', 'y2', 'y3']
    target_names = {
        'y1': 'y1 (模型回覆安全性預測)',
        'y2': 'y2 (提示詞有害性預測)',
        'y3': 'y3 (安全判定一致性預測)'
    }
    
    # Prepare list for storing all metrics
    metrics_list = []
    
    # For drawing layer-wise curves later
    # Format: {(target, model, layer, scheme, dataset): ECE}
    ece_trends = {}
    
    # Loop targets and layers
    for target_idx, target_name in enumerate(targets):
        y_train = y_targets_train[target_idx]
        y_eval = y_targets_eval[target_idx]
        
        # Calculate target positive class ratio from eval
        target_ratio_pos = np.mean(y_eval)
        print("\n" + "="*70)
        print(f"處理目標變數: {target_names[target_name]}")
        print(f"  └─ Eval 集中正標記(1)比例 = {target_ratio_pos:.4f} (負標記(0)比例 = {1.0 - target_ratio_pos:.4f})")
        print("="*70)
        
        for layer_idx in range(num_layers):
            layer_num = layer_idx + 1
            print(f"\n--- [第 {layer_num} / {num_layers} 層隱藏狀態] ---")
            
            X_2d_train = X_3d_train[:, layer_idx, :]
            X_2d_eval = X_3d_eval[:, layer_idx, :]
            
            # Obtain standard splits from df_10000
            _, _, X_test, _, _, y_test, _ = DataSplitter.split_and_scale(X_2d_train, y_train, layer_idx)
            
            # test1 (calibration) and test2 (test)
            X_test1, X_test2, y_test1, y_test2 = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
            y_test1_np, y_test2_np, y_eval_np = np.array(y_test1), np.array(y_test2), np.array(y_eval)
            
            # Extract pool indices for the current target and reset pool for this layer
            pool_idx_1 = df_unused[df_unused[target_name] == 1].index.tolist()
            pool_idx_0 = df_unused[df_unused[target_name] == 0].index.tolist()
            
            # Supplement test1 and test2 to match eval positive class ratio using df_unused
            X_test1_adj, y_test1_adj = supplement_to_ratio(
                X_test1, y_test1_np, df_unused, pool_idx_1, pool_idx_0, 
                target_ratio_pos, target_name, layer_idx, random_state=42
            )
            
            X_test2_adj, y_test2_adj = supplement_to_ratio(
                X_test2, y_test2_np, df_unused, pool_idx_1, pool_idx_0, 
                target_ratio_pos, target_name, layer_idx, random_state=43
            )
            
            print(f"  ▶ 樣本數量變化 (資料補充法):")
            print(f"    ├─ Test1 (校正集): {len(y_test1_np)} ➔ 補充後: {len(y_test1_adj)} 筆 (正標記比: {np.mean(y_test1_adj):.4f})")
            print(f"    ├─ Test2 (測試集): {len(y_test2_np)} ➔ 補充後: {len(y_test2_adj)} 筆 (正標記比: {np.mean(y_test2_adj):.4f})")
            print(f"    └─ Eval  (驗證集): {len(y_eval_np)} 筆 (正標記比: {np.mean(y_eval_np):.4f})")
            
            # Setup layer directory for models
            model_save_layer_dir = os.path.join(output_dir, f"layer_{layer_num}")
            os.makedirs(model_save_layer_dir, exist_ok=True)
            
            # Dictionary to store reliability curves data for plotting
            reliability_plot_data_test2 = {}
            reliability_plot_data_eval = {}
            
            # Loop models
            for model_name in models:
                # Load pre-trained best classifier
                clf_path = f"results/unified_training/layer_{layer_num}/{model_name.lower()}_{target_name.lower()}_best.pkl"
                if not os.path.exists(clf_path):
                    continue
                
                clf = joblib.load(clf_path)
                
                # Setup predictions
                # 1. Raw Predictions
                prob_raw_test2 = clf.predict_proba(X_test2)[:, 1]
                prob_raw_test2_adj = clf.predict_proba(X_test2_adj)[:, 1]
                prob_raw_eval = clf.predict_proba(X_2d_eval)[:, 1]
                
                # 2. Standard Calibration (Isotonic fitted on original test1)
                cal_std = get_calibrated_classifier(clf, method='isotonic')
                cal_std.fit(X_test1, y_test1_np)
                
                prob_std_test2 = cal_std.predict_proba(X_test2)[:, 1]
                prob_std_test2_adj = cal_std.predict_proba(X_test2_adj)[:, 1]
                prob_std_eval = cal_std.predict_proba(X_2d_eval)[:, 1]
                
                # 3. Ratio-Adjusted Calibration (Isotonic fitted on test1_adj)
                cal_adj = get_calibrated_classifier(clf, method='isotonic')
                cal_adj.fit(X_test1_adj, y_test1_adj)
                
                prob_adj_test2_adj = cal_adj.predict_proba(X_test2_adj)[:, 1]
                prob_adj_eval = cal_adj.predict_proba(X_2d_eval)[:, 1]
                
                # Save the new ratio-adjusted calibrated model
                model_name_cal = f"{model_name.lower()}_{target_name.lower()}_calibrated.pkl"
                joblib.dump(cal_adj, os.path.join(model_save_layer_dir, model_name_cal))
                
                # Store curve data
                reliability_plot_data_test2[model_name] = {
                    'Raw': (y_test2_adj, prob_raw_test2_adj),
                    'StdCal': (y_test2_adj, prob_std_test2_adj),
                    'AdjCal': (y_test2_adj, prob_adj_test2_adj)
                }
                reliability_plot_data_eval[model_name] = {
                    'Raw': (y_eval_np, prob_raw_eval),
                    'StdCal': (y_eval_np, prob_std_eval),
                    'AdjCal': (y_eval_np, prob_adj_eval)
                }
                
                # Record metrics
                eval_configs = [
                    # Dataset, Scheme, y_true, y_prob
                    ('test2', 'Raw', y_test2_np, prob_raw_test2),
                    ('test2', 'StdCal', y_test2_np, prob_std_test2),
                    ('test2_adj', 'Raw', y_test2_adj, prob_raw_test2_adj),
                    ('test2_adj', 'StdCal', y_test2_adj, prob_std_test2_adj),
                    ('test2_adj', 'AdjCal', y_test2_adj, prob_adj_test2_adj),
                    ('eval', 'Raw', y_eval_np, prob_raw_eval),
                    ('eval', 'StdCal', y_eval_np, prob_std_eval),
                    ('eval', 'AdjCal', y_eval_np, prob_adj_eval)
                ]
                
                print(f"    ├─ [{model_name} 模型評估]")
                for ds_name, scheme, y_t, y_p in eval_configs:
                    brier = brier_score_loss(y_t, y_p)
                    ece = calculate_ece(y_t, y_p)
                    loss_val = log_loss(y_t, y_p)
                    
                    metrics_list.append({
                        'target': target_name,
                        'layer': layer_num,
                        'model': model_name,
                        'dataset': ds_name,
                        'scheme': scheme,
                        'brier': brier,
                        'ece': ece,
                        'logloss': loss_val
                    })
                    
                    ece_trends[(target_name, model_name, layer_num, scheme, ds_name)] = ece
                    
                # Print output summary for important schemes
                print(f"      ▶ Raw (test2_adj)     - ECE: {ece_trends[(target_name, model_name, layer_num, 'Raw', 'test2_adj')]:.4f} | Brier: {brier_score_loss(y_test2_adj, prob_raw_test2_adj):.4f}")
                print(f"      ▶ StdCal (test2_adj)  - ECE: {ece_trends[(target_name, model_name, layer_num, 'StdCal', 'test2_adj')]:.4f} | Brier: {brier_score_loss(y_test2_adj, prob_std_test2_adj):.4f}")
                print(f"      ▶ AdjCal (test2_adj)  - ECE: {ece_trends[(target_name, model_name, layer_num, 'AdjCal', 'test2_adj')]:.4f} | Brier: {brier_score_loss(y_test2_adj, prob_adj_test2_adj):.4f}")
                print(f"      ▶ StdCal (eval)       - ECE: {ece_trends[(target_name, model_name, layer_num, 'StdCal', 'eval')]:.4f} | Brier: {brier_score_loss(y_eval_np, prob_std_eval):.4f}")
                print(f"      ▶ AdjCal (eval)       - ECE: {ece_trends[(target_name, model_name, layer_num, 'AdjCal', 'eval')]:.4f} | Brier: {brier_score_loss(y_eval_np, prob_adj_eval):.4f}")
                print("      └───────────────────────────────────────────────────")
            
            # Generate Reliability curves for this Layer + Target
            # 1. On adjusted test2 (ID supplemented)
            t2_plot_name = f"reliability_diagram_test2_adj_{target_name}_layer{layer_num}.png"
            generate_individual_reliability_plot(
                reliability_plot_data_test2,
                f"Layer {layer_num} - {target_names[target_name]} 信賴度曲線 [Test2 比例補充集]",
                os.path.join(plots_dir, t2_plot_name)
            )
            
            # 2. On eval (OOD)
            eval_plot_name = f"reliability_diagram_eval_{target_name}_layer{layer_num}.png"
            generate_individual_reliability_plot(
                reliability_plot_data_eval,
                f"Layer {layer_num} - {target_names[target_name]} 信賴度曲線 [Eval 外部驗證集]",
                os.path.join(plots_dir, eval_plot_name)
            )
            
    # Save metrics data to CSV
    df_metrics = pd.DataFrame(metrics_list)
    csv_path = os.path.join(output_dir, "adjusted_ratio_metrics.csv")
    df_metrics.to_csv(csv_path, index=False)
    print(f"\n📊 實驗結果指標數據已保存至: {csv_path}")
    
    # ----------------------------------------------------
    # 畫重要圖 A: ECE & Brier Score 對比長條圖 (Layer 6, target y2)
    # ----------------------------------------------------
    print("\n[繪製圖表 A] ECE/Brier Score 方案對比圖...")
    setup_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    df_sub = df_metrics[(df_metrics['target'] == 'y2') & (df_metrics['layer'] == 6)]
    
    # Selected evaluation scenarios
    scenarios = [
        ('test2_adj', 'Raw', 'Raw (test2_adj)', '#7F7F7F'),
        ('test2_adj', 'StdCal', 'StdCal (test2_adj)', '#1F77B4'),
        ('test2_adj', 'AdjCal', 'AdjCal (test2_adj)', '#FF7F0E'),
        ('eval', 'StdCal', 'StdCal (eval)', '#2CA02C'),
        ('eval', 'AdjCal', 'AdjCal (eval)', '#D62728'),
    ]
    
    x = np.arange(len(models))
    width = 0.15
    
    # Plot ECE on left ax, Brier Score on right ax
    for i, (ds_name, scheme, label, color) in enumerate(scenarios):
        ece_vals = []
        brier_vals = []
        for model in models:
            row = df_sub[(df_sub['model'] == model) & (df_sub['dataset'] == ds_name) & (df_sub['scheme'] == scheme)]
            ece_vals.append(row['ece'].values[0] if not row.empty else 0)
            brier_vals.append(row['brier'].values[0] if not row.empty else 0)
            
        axes[0].bar(x + i*width - 2*width, ece_vals, width, label=label, color=color, edgecolor='black', linewidth=0.5)
        axes[1].bar(x + i*width - 2*width, brier_vals, width, label=label, color=color, edgecolor='black', linewidth=0.5)
        
    axes[0].set_title("第 6 層 - 提示詞有害性預測 (y2) ECE 對比 (資料補充法)\n(校正誤差，越小越好)", fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].set_ylabel("Expected Calibration Error (ECE)")
    axes[0].grid(True, axis='y', linestyle='--', alpha=0.3)
    axes[0].legend()
    
    axes[1].set_title("第 6 層 - 提示詞有害性預測 (y2) Brier Score 對比 (資料補充法)\n(整體校正度與分類誤差，越小越好)", fontsize=13, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models)
    axes[1].set_ylabel("Brier Score")
    axes[1].grid(True, axis='y', linestyle='--', alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    bar_save_path = os.path.join(plots_dir, "metric_comparison_y2_layer6.png")
    plt.savefig(bar_save_path, dpi=150)
    plt.close()
    print(f"  └─ 已儲存至: {bar_save_path}")
    
    # ----------------------------------------------------
    # 畫重要圖 B: Layer-wise ECE 趨勢折線圖 (Target y2)
    # ----------------------------------------------------
    print("\n[繪製圖表 B] Layer-wise ECE 趨勢折線圖...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes_flat = axes.flatten()
    
    layers = list(range(1, num_layers + 1))
    
    line_scenarios = [
        ('test2', 'StdCal', 'StdCal on test2 (ID)', 'o', '#1F77B4', '-'),
        ('test2_adj', 'StdCal', 'StdCal on test2_adj (Shifted ID)', 's', '#AEC7E8', '--'),
        ('test2_adj', 'AdjCal', 'AdjCal on test2_adj (Shifted ID)', '^', '#FF7F0E', '-'),
        ('eval', 'StdCal', 'StdCal on eval (OOD)', 'x', '#2CA02C', '--'),
        ('eval', 'AdjCal', 'AdjCal on eval (OOD)', 'd', '#D62728', '-'),
    ]
    
    for idx, model_name in enumerate(models):
        ax = axes_flat[idx]
        
        for ds_name, scheme, label, marker, color, linestyle in line_scenarios:
            vals = []
            for l in layers:
                val = ece_trends.get(( 'y2', model_name, l, scheme, ds_name ), None)
                vals.append(val)
                
            ax.plot(layers, vals, marker=marker, linestyle=linestyle, color=color, label=label, linewidth=1.8, markersize=7)
            
        ax.set_title(f"{model_name} 模型 (y2 目標)", fontsize=13, fontweight='bold')
        ax.set_xlabel("隱藏特徵層數 (Layer)")
        ax.set_ylabel("Expected Calibration Error (ECE)")
        ax.set_xticks(layers)
        ax.grid(True, linestyle='--', alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=9, loc='upper right')
            
    # Hide the empty 6th subplot
    axes_flat[5].axis('off')
    
    fig.suptitle("提示詞有害性預測 (y2) 跨層 ECE 變化趨勢對比 (資料補充法)", fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    trend_save_path = os.path.join(plots_dir, "layerwise_ece_trends_y2.png")
    plt.savefig(trend_save_path, dpi=150)
    plt.close()
    print(f"  └─ 已儲存至: {trend_save_path}")
    
    # Close logger
    sys.stdout = sys.__stdout__
    logger.close()
    print("\n[OK] 比例調整與校正實驗圓滿完成！")
    print(f"[OK] 所有結果已儲存至: {output_dir}")

if __name__ == "__main__":
    main()
