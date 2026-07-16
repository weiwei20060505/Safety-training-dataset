import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.calibration import IsotonicRegression
from sklearn.linear_model import LogisticRegression

# Import from project modules
from unified_train import DataPreprocessor
import utils_calibration

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

def plot_calibration_curves(plot_data, layer_num, dtype_name, save_path):
    """
    Plots a 2x3 grid showing calibration curves (reliability diagrams).
    Columns: Target y1, Target y2, Target y3
    Row 0: Isotonic Calibration (Raw dashed line vs Calibrated solid line for 5 models)
    Row 1: Logistic Platt Calibration (Raw dashed line vs Calibrated solid line for 5 models)
    """
    utils_calibration.setup_chinese_font()
    models = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
    colors = {'SGD': '#4C72B0', 'MLP': '#55A868', 'LGB': '#C44E52', 'LR': '#8172B3', 'RF': '#CCB974'}
    markers = {'SGD': 'o', 'MLP': 's', 'LGB': '^', 'LR': 'v', 'RF': 'D'}
    bin_edges = np.linspace(0.0, 1.0, 11)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), sharex=False, sharey=True)
    fig.suptitle(f"第 {layer_num} 層隱藏狀態 - 三種目標任務之校正對比曲線 (數據組: {dtype_name} | Test2)", fontsize=16, fontweight='bold')
    
    targets_list = ['y1', 'y2', 'y3']
    target_titles = {
        'y1': '任務 y1: 模型回覆安全性\n(P(y1==y1gt|X), y3gt)',
        'y2': '任務 y2: 提示詞有害性\n(P(y2==y1gt|X), y3gt)',
        'y3': '任務 y3: 一致性預測\n(P(y3==y3gt|X), y3gt)'
    }
    
    for t_idx, t_key in enumerate(targets_list):
        ax_iso = axes[0, t_idx]
        ax_iso.plot([0, 1], [0, 1], "k--", label="完美校正線", alpha=0.5)
        
        ax_lr = axes[1, t_idx]
        ax_lr.plot([0, 1], [0, 1], "k--", label="完美校正線", alpha=0.5)
        
        for model_name in models:
            if t_key not in plot_data or model_name not in plot_data[t_key]:
                continue
            d = plot_data[t_key][model_name]
            
            # Raw vs Isotonic (Row 0)
            frac_pos_raw, mean_pred_raw, _ = utils_calibration.calculate_calibration_curve(d['y_true_test2'], d['y_prob_raw_test2'], bin_edges)
            frac_pos_iso, mean_pred_iso, _ = utils_calibration.calculate_calibration_curve(d['y_true_test2'], d['y_prob_iso_test2'], bin_edges)
            
            raw_brier = utils_calibration.calculate_all_metrics(d['y_true_test2'], d['y_prob_raw_test2'])['brier']
            iso_brier = utils_calibration.calculate_all_metrics(d['y_true_test2'], d['y_prob_iso_test2'])['brier']
            
            # Raw (dashed)
            ax_iso.plot(mean_pred_raw, frac_pos_raw, linestyle='--', marker=markers[model_name], markersize=4,
                        color=colors[model_name], alpha=0.4, label=f"{model_name} Raw (Brier: {raw_brier:.3f})")
            # Isotonic (solid)
            ax_iso.plot(mean_pred_iso, frac_pos_iso, linestyle='-', marker=markers[model_name], markersize=6,
                        color=colors[model_name], linewidth=2.0, label=f"{model_name} Cal (Brier: {iso_brier:.3f})")
            
            # Raw vs Logistic (Row 1)
            frac_pos_lr, mean_pred_lr, _ = utils_calibration.calculate_calibration_curve(d['y_true_test2'], d['y_prob_lr_test2'], bin_edges)
            lr_brier = utils_calibration.calculate_all_metrics(d['y_true_test2'], d['y_prob_lr_test2'])['brier']
            
            # Raw (dashed)
            ax_lr.plot(mean_pred_raw, frac_pos_raw, linestyle='--', marker=markers[model_name], markersize=4,
                       color=colors[model_name], alpha=0.4, label=f"{model_name} Raw (Brier: {raw_brier:.3f})")
            # Logistic (solid)
            ax_lr.plot(mean_pred_lr, frac_pos_lr, linestyle='-', marker=markers[model_name], markersize=6,
                       color=colors[model_name], linewidth=2.0, label=f"{model_name} Cal (Brier: {lr_brier:.3f})")
                       
        ax_iso.set_xlim([-0.05, 1.05])
        ax_iso.set_ylim([-0.05, 1.05])
        ax_iso.set_title(f"{target_titles[t_key]}\n[Isotonic 校正]", fontsize=12, fontweight='bold')
        ax_iso.grid(True, linestyle='--', alpha=0.3)
        ax_iso.legend(loc="upper left", fontsize=7.5, framealpha=0.7)
        if t_idx == 0:
            ax_iso.set_ylabel("實際正類比例", fontsize=11)
            
        ax_lr.set_xlim([-0.05, 1.05])
        ax_lr.set_ylim([-0.05, 1.05])
        ax_lr.set_title(f"[Logistic Platt 校正]", fontsize=12, fontweight='bold')
        ax_lr.set_xlabel("平均預測機率 / 置信度 (X 軸)", fontsize=11)
        ax_lr.grid(True, linestyle='--', alpha=0.3)
        ax_lr.legend(loc="upper left", fontsize=7.5, framealpha=0.7)
        if t_idx == 0:
            ax_lr.set_ylabel("實際正類比例", fontsize=11)
            
    plt.tight_layout()
    fig.subplots_adjust(top=0.90)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def generate_comparison_grid(df_metrics, data_type_name, dataset_name, title, save_path):
    """
    Plots a 3x2 grid comparing SGD, MLP, LGB, LR, RF across layers.
    Columns: Brier Score, Log Loss
    Rows: Target y1, Target y2, Target y3
    """
    utils_calibration.setup_chinese_font()
    models = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
    colors = {'SGD': '#4C72B0', 'MLP': '#55A868', 'LGB': '#C44E52', 'LR': '#8172B3', 'RF': '#CCB974'}
    markers = {'SGD': 'o', 'MLP': 's', 'LGB': '^', 'LR': 'v', 'RF': 'D'}
    
    targets = ['y1', 'y2', 'y3']
    target_titles = {
        'y1': '任務 y1: (P(y1==y1gt|X), y3gt)',
        'y2': '任務 y2: (P(y2==y1gt|X), y3gt)',
        'y3': '任務 y3: (P(y3==y3gt|X), y3gt)'
    }
    metrics = ['Brier Score', 'Log Loss']
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 15), sharex=True)
    fig.suptitle(f'{title}\n[數據類型: {data_type_name} | 評估集: {dataset_name}]', fontsize=16, fontweight='bold', y=0.96)
    
    for r_idx, t_id in enumerate(targets):
        for c_idx, mt in enumerate(metrics):
            ax = axes[r_idx, c_idx]
            
            df_sub = df_metrics[(df_metrics['version'] == t_id) & (df_metrics['metric'] == mt)]
            
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
                ax.set_title(f'{mt}', fontsize=13, fontweight='bold')
            if c_idx == 0:
                ax.set_ylabel(f'{target_titles[t_id]}\n指標值 (越低越好)', fontsize=11, fontweight='bold')
            if r_idx == 2:
                ax.set_xlabel('隱藏狀態特徵層數 (Layer)', fontsize=11)
                
            num_layers = int(df_metrics['layer'].max()) if not df_metrics.empty else 6
            ax.set_xticks(range(1, num_layers + 1))
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # Remove duplicate labels in legend
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8.5)
            
    plt.tight_layout()
    fig.subplots_adjust(top=0.90)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def main():
    base_output_dir = "results/calibration_comparison"
    model_dirs = {
        'Augmented': os.path.join(base_output_dir, "model_aug"),
        'Aligned': os.path.join(base_output_dir, "model_align")
    }
    for d in model_dirs.values():
        os.makedirs(d, exist_ok=True)
        
    sys.stdout = DualLogger(os.path.join(base_output_dir, "calibration_comparison_log.txt"))
    
    print("="*80)
    print(" 啟動 Step 6: 調整後數據（擴增與對齊）之三種目標任務校正對比與多維指標繪圖")
    print("="*80)
    
    # Paths
    AUG_TEST1_PATH = "augmented_test1.pkl"
    AUG_TEST2_PATH = "augmented_test2.pkl"
    ALIGN_TEST1_PATH = "aligned_test1.pkl"
    ALIGN_TEST2_PATH = "aligned_test2.pkl"
    EVAL_PATH = "experiment_results_eval.pkl"
    
    # Check paths
    for p in [AUG_TEST1_PATH, AUG_TEST2_PATH, ALIGN_TEST1_PATH, ALIGN_TEST2_PATH, EVAL_PATH]:
        if not os.path.exists(p):
            print(f"錯誤: 確保 {p} 檔案存在。請先運行 run_02。")
            sys.exit(1)
            
    import gc
    print("[1] 載入資料集與抽取特徵...")
    
    print("  └─ 載入 augmented_test1...")
    aug_test1_dict = joblib.load(AUG_TEST1_PATH)
    
    print("  └─ 載入 augmented_test2...")
    aug_test2_dict = joblib.load(AUG_TEST2_PATH)
    
    print("  └─ 載入 aligned_test1...")
    align_test1_dict = joblib.load(ALIGN_TEST1_PATH)
    
    print("  └─ 載入 aligned_test2...")
    align_test2_dict = joblib.load(ALIGN_TEST2_PATH)
    
    print("  └─ 載入 eval...")
    prep_eval = DataPreprocessor(EVAL_PATH)
    prep_eval.load_data()
    X_3d_eval = prep_eval.extract_features()
    y_y1_eval, y_y2_eval, _ = prep_eval.create_targets()
    y1_eval_vals = y_y1_eval.values
    y3_eval_vals = (y_y1_eval == y_y2_eval).astype(int).values
    del prep_eval, y_y1_eval, y_y2_eval
    gc.collect()
    
    num_layers = X_3d_eval.shape[1]
    models = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
    
    # Data type configurations
    data_types = {
        'Augmented': {
            'test1_dict': aug_test1_dict,
            'test2_dict': aug_test2_dict,
            'test1_name': 'data_aug_test1', 'test2_name': 'data_aug_test2'
        },
        'Aligned': {
            'test1_dict': align_test1_dict,
            'test2_dict': align_test2_dict,
            'test1_name': 'data_align_test1', 'test2_name': 'data_align_test2'
        }
    }
    
    for dtype, d_setup in data_types.items():
        print("\n" + "="*75)
        print(f"執行校正組：{dtype} 數據集")
        print("="*75)
        
        target_dir = model_dirs[dtype]
        records = []
        
        test1_dict = d_setup['test1_dict']
        test2_dict = d_setup['test2_dict']
        test1_name = d_setup['test1_name']
        test2_name = d_setup['test2_name']
        
        for layer_idx in range(num_layers):
            layer_num = layer_idx + 1
            print(f"  ├─ 正在計算第 {layer_num} 層...")
            
            X_ev_layer = X_3d_eval[:, layer_idx, :]
            
            layer_plot_data = {
                'y1': {},
                'y2': {},
                'y3': {}
            }
            
            for model_name in models:
                path_y1 = f"results/unified_training/layer_{layer_num}/{model_name.lower()}_y1_best.pkl"
                path_y2 = f"results/unified_training/layer_{layer_num}/{model_name.lower()}_y2_best.pkl"
                path_y3 = f"results/unified_training/layer_{layer_num}/{model_name.lower()}_y3_best.pkl"
                
                if not (os.path.exists(path_y1) and os.path.exists(path_y2) and os.path.exists(path_y3)):
                    continue
                    
                clf_y1 = joblib.load(path_y1)
                clf_y2 = joblib.load(path_y2)
                clf_y3 = joblib.load(path_y3)
                
                # We loop over the three targets (y1, y2, y3 tasks)
                targets_loop = [
                    ('y1', clf_y1),
                    ('y2', clf_y2),
                    ('y3', clf_y3)
                ]
                
                for target_key, clf in targets_loop:
                    # Load separate DataFrames since train_test_split is target-specific
                    df_t1 = test1_dict[target_key]
                    df_t2 = test2_dict[target_key]
                    
                    X_t1_layer = np.array(df_t1['hidden_state'].tolist())[:, layer_idx, :]
                    X_t2_layer = np.array(df_t2['hidden_state'].tolist())[:, layer_idx, :]
                    
                    y1_t1 = df_t1['y1'].values
                    y1_t2 = df_t2['y1'].values
                    
                    y3_t1 = df_t1['y3'].values
                    y3_t2 = df_t2['y3'].values
                    
                    # Predict raw probs
                    p_t1 = clf.predict_proba(X_t1_layer)[:, 1]
                    p_t2 = clf.predict_proba(X_t2_layer)[:, 1]
                    p_ev = clf.predict_proba(X_ev_layer)[:, 1]
                    
                    # Construct calibration features and target labels
                    if target_key == 'y1':
                        X_tr_cal = p_t1 * y1_t1 + (1.0 - p_t1) * (1.0 - y1_t1)
                        X_te_cal = p_t2 * y1_t2 + (1.0 - p_t2) * (1.0 - y1_t2)
                        X_ev_cal = p_ev * y1_eval_vals + (1.0 - p_ev) * (1.0 - y1_eval_vals)
                        Y_tr = y3_t1
                        Y_te = y3_t2
                        Y_ev = y3_eval_vals
                    elif target_key == 'y2':
                        X_tr_cal = p_t1 * y1_t1 + (1.0 - p_t1) * (1.0 - y1_t1)
                        X_te_cal = p_t2 * y1_t2 + (1.0 - p_t2) * (1.0 - y1_t2)
                        X_ev_cal = p_ev * y1_eval_vals + (1.0 - p_ev) * (1.0 - y1_eval_vals)
                        Y_tr = y3_t1
                        Y_te = y3_t2
                        Y_ev = y3_eval_vals
                    else: # y3
                        X_tr_cal = p_t1 * y3_t1 + (1.0 - p_t1) * (1.0 - y3_t1)
                        X_te_cal = p_t2 * y3_t2 + (1.0 - p_t2) * (1.0 - y3_t2)
                        X_ev_cal = p_ev * y3_eval_vals + (1.0 - p_ev) * (1.0 - y3_eval_vals)
                        Y_tr = y3_t1
                        Y_te = y3_t2
                        Y_ev = y3_eval_vals
                        
                    # 1. Raw Metrics
                    raw_test2 = utils_calibration.calculate_all_metrics(Y_te, X_te_cal)
                    raw_eval = utils_calibration.calculate_all_metrics(Y_ev, X_ev_cal)
                    
                    # 2. Isotonic Calibration
                    iso = IsotonicRegression(out_of_bounds='clip')
                    iso.fit(X_tr_cal, Y_tr)
                    p_iso_test1 = iso.predict(X_tr_cal)
                    p_iso_test2 = iso.predict(X_te_cal)
                    p_iso_eval = iso.predict(X_ev_cal)
                    
                    iso_test1 = utils_calibration.calculate_all_metrics(Y_tr, p_iso_test1)
                    iso_test2 = utils_calibration.calculate_all_metrics(Y_te, p_iso_test2)
                    iso_eval = utils_calibration.calculate_all_metrics(Y_ev, p_iso_eval)
                    
                    # 3. Logistic Platt Calibration
                    lr = LogisticRegression()
                    lr.fit(X_tr_cal.reshape(-1, 1), Y_tr)
                    p_lr_test1 = lr.predict_proba(X_tr_cal.reshape(-1, 1))[:, 1]
                    p_lr_test2 = lr.predict_proba(X_te_cal.reshape(-1, 1))[:, 1]
                    p_lr_eval = lr.predict_proba(X_ev_cal.reshape(-1, 1))[:, 1]
                    
                    iso_lr_test1 = utils_calibration.calculate_all_metrics(Y_tr, p_lr_test1)
                    lr_test2 = utils_calibration.calculate_all_metrics(Y_te, p_lr_test2)
                    lr_eval = utils_calibration.calculate_all_metrics(Y_ev, p_lr_eval)
                    
                    # Print results to log (without ECE)
                    print(f"    [{model_name} - Layer {layer_num} - Target {target_key}]")
                    print(f"      Test1 => Raw Brier: {utils_calibration.calculate_all_metrics(Y_tr, X_tr_cal)['brier']:.5f} | Iso Brier: {iso_test1['brier']:.5f} | LR Brier: {iso_lr_test1['brier']:.5f}")
                    print(f"      Test2 => Raw Brier: {raw_test2['brier']:.5f} | Iso Brier: {iso_test2['brier']:.5f} | LR Brier: {lr_test2['brier']:.5f}")
                    print(f"      Eval  => Raw Brier: {raw_eval['brier']:.5f} | Iso Brier: {iso_eval['brier']:.5f} | LR Brier: {lr_eval['brier']:.5f}")
                    
                    # Append records for test1
                    for metric_name, metric_val in [('Brier Score', iso_test1['brier']), ('Log Loss', iso_test1['logloss'])]:
                        records.append({
                            'version': target_key, # lowercase key to match targets in generate_comparison_grid!
                            'layer': layer_num,
                            'model': model_name,
                            'dataset': test1_name,
                            'metric': metric_name,
                            'value': metric_val
                        })
                    # Append records for test2
                    for metric_name, metric_val in [('Brier Score', iso_test2['brier']), ('Log Loss', iso_test2['logloss'])]:
                        records.append({
                            'version': target_key,
                            'layer': layer_num,
                            'model': model_name,
                            'dataset': test2_name,
                            'metric': metric_name,
                            'value': metric_val
                        })
                    # Append records for eval
                    for metric_name, metric_val in [('Brier Score', iso_eval['brier']), ('Log Loss', iso_eval['logloss'])]:
                        records.append({
                            'version': target_key,
                            'layer': layer_num,
                            'model': model_name,
                            'dataset': 'data_eval',
                            'metric': metric_name,
                            'value': metric_val
                        })
                        
                    # Save calibration plot data
                    layer_plot_data[target_key][model_name] = {
                        'y_true_test2': Y_te,
                        'y_prob_raw_test2': X_te_cal,
                        'y_prob_iso_test2': p_iso_test2,
                        'y_prob_lr_test2': p_lr_test2
                    }
                    
            # Plot calibration curves for this layer
            save_path_cal = os.path.join(target_dir, f"Layer-{layer_num}_AllTargets-y1-y2-y3_CalibrationCurves_{dtype}.png")
            plot_calibration_curves(layer_plot_data, layer_num, dtype, save_path_cal)
            print(f"    └─ 成功生成校正曲線對比圖至: {save_path_cal}")
            
        # Generate grid plots (save inside target_dir)
        df_m = pd.DataFrame(records)
        
        save_plot_test1 = os.path.join(target_dir, "metrics_comparison_test1.png")
        generate_comparison_grid(df_m[df_m['dataset'] == test1_name], dtype, test1_name, 
                                 f"三目標指標對比圖 (Isotonic 校正 - {dtype} Test1)", save_plot_test1)
        print(f"  └─ 成功保存 {dtype} Test1 折線對比圖至: {save_plot_test1}")
        
        save_plot_test2 = os.path.join(target_dir, "metrics_comparison_test2.png")
        generate_comparison_grid(df_m[df_m['dataset'] == test2_name], dtype, test2_name, 
                                 f"三目標指標對比圖 (Isotonic 校正 - {dtype} Test2)", save_plot_test2)
        print(f"  └─ 成功保存 {dtype} Test2 折線對比圖至: {save_plot_test2}")
        
        save_plot_eval = os.path.join(target_dir, "metrics_comparison_eval.png")
        generate_comparison_grid(df_m[df_m['dataset'] == 'data_eval'], dtype, 'data_eval', 
                                 f"三目標指標對比圖 (Isotonic 校正 - {dtype} Eval)", save_plot_eval)
        print(f"  └─ 成功保存 {dtype} Eval 折線對比圖至: {save_plot_eval}")
        
    print("\n" + "="*80)
    print(" Step 6: 運行完成！已將 Test1, Test2 與 Eval 指標圖分類保存於 model_aug/ 和 model_align/。")
    print("="*80)

if __name__ == "__main__":
    main()
