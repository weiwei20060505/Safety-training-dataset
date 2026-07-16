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
from sklearn.metrics import brier_score_loss, log_loss

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

def print_bin_details(y_true, y_prob, edges):
    """Generates detailed text of binning for the logs."""
    bin_ids = np.digitize(y_prob, edges)
    lines = []
    for idx in range(len(edges) - 1):
        mask = (bin_ids == idx + 1)
        n_samples = np.sum(mask)
        if n_samples > 0:
            mean_pred = np.mean(y_prob[mask])
            frac_pos = np.mean(y_true[mask])
            bias = np.abs(mean_pred - frac_pos)
            lines.append(f"      ├─ Bin {idx+1} [{edges[idx]:.1f}, {edges[idx+1]:.1f}]: 樣本數={n_samples:5d} | 平均預測={mean_pred:.4f} | 實際正類率={frac_pos:.4f} | 偏差={bias:.4f}")
        else:
            lines.append(f"      ├─ Bin {idx+1} [{edges[idx]:.1f}, {edges[idx+1]:.1f}]: 樣本數=    0 | 無數據")
    return "\n".join(lines)

def main():
    base_output_dir = "results/three_versions_calibration"
    os.makedirs(base_output_dir, exist_ok=True)
    sys.stdout = DualLogger(os.path.join(base_output_dir, "calibration_detailed_log.txt"))
    
    print("="*80)
    print(" 啟動 Step 5: 三種校正資料組模型校正評估與視覺化分析管線")
    print("="*80)
    
    TRAIN_PATH = "experiment_results_train_10000.pkl"
    EVAL_PATH = "experiment_results_eval.pkl"
    
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(EVAL_PATH):
        print(f"錯誤: 確保 {TRAIN_PATH} 與 {EVAL_PATH} 存在。")
        sys.exit(1)
        
    print(f"[1] 載入訓練集: {TRAIN_PATH}")
    prep_train = DataPreprocessor(TRAIN_PATH)
    prep_train.load_data()
    X_3d_train = prep_train.extract_features()
    y_y1_train, y_y2_train, _ = prep_train.create_targets()
    y_y3_train = (y_y1_train == y_y2_train).astype(int)
    
    print(f"[2] 載入外部評估集: {EVAL_PATH}")
    prep_eval = DataPreprocessor(EVAL_PATH)
    prep_eval.load_data()
    X_3d_eval = prep_eval.extract_features()
    y_y1_eval, y_y2_eval, _ = prep_eval.create_targets()
    y_y3_eval = (y_y1_eval == y_y2_eval).astype(int)
    
    num_layers = X_3d_train.shape[1]
    models = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
    
    # Define uniform bin edges [0.0, 1.0] for plotting and bin logging
    bin_edges = np.linspace(0.0, 1.0, 11)
    
    # Store global summaries for the final markdown report in log
    global_summaries = []
    
    for layer_idx in range(num_layers):
        layer_num = layer_idx + 1
        print("\n" + "="*75)
        print(f"處理第 {layer_num} / {num_layers} 層隱藏狀態")
        print("="*75)
        
        # Features for this layer
        X_2d_train = X_3d_train[:, layer_idx, :]
        X_2d_eval = X_3d_eval[:, layer_idx, :]
        
        # Split logic exactly identical to run_01 to keep test1 and test2 consistent
        indices = np.arange(len(y_y1_train))
        idx_train_val, idx_test = train_test_split(indices, test_size=0.2, random_state=42, stratify=y_y1_train)
        idx_train, idx_val = train_test_split(idx_train_val, test_size=0.25, random_state=42, stratify=y_y1_train[idx_train_val])
        idx_test1, idx_test2 = train_test_split(idx_test, test_size=0.5, random_state=42)
        
        # Extract features and targets for test1, test2, and eval
        X_test1 = X_2d_train[idx_test1]
        X_test2 = X_2d_train[idx_test2]
        
        y1_test1 = y_y1_train.values[idx_test1]
        y1_test2 = y_y1_train.values[idx_test2]
        y1_eval = y_y1_eval.values
        
        y3_test1 = y_y3_train.values[idx_test1]
        y3_test2 = y_y3_train.values[idx_test2]
        y3_eval = y_y3_eval.values
        
        # Dictionary to hold plotting data
        plot_data = {
            'v1': {},
            'v2': {},
            'v3': {}
        }
        
        for model_name in models:
            model_path = f"results/unified_training/layer_{layer_num}/{model_name.lower()}_y1_best.pkl"
            if not os.path.exists(model_path):
                print(f"  [跳過] 未找到模型 {model_name} 在第 {layer_num} 層的安全探針。")
                continue
                
            clf = joblib.load(model_path)
            
            # Predict p1 (Unsafe probability)
            p1_test1 = clf.predict_proba(X_test1)[:, 1]
            p1_test2 = clf.predict_proba(X_test2)[:, 1]
            p1_eval = clf.predict_proba(X_2d_eval)[:, 1]
            
            # Helper to train and evaluate 1D calibration
            def run_calibration_for_pair(X_tr, Y_tr, X_te, Y_te, X_ev, Y_ev, version_id, version_name):
                # 1. Raw Metrics
                raw_metrics_test2 = utils_calibration.calculate_all_metrics(Y_te, X_te)
                raw_metrics_eval = utils_calibration.calculate_all_metrics(Y_ev, X_ev)
                
                # 2. Isotonic Calibration
                iso = IsotonicRegression(out_of_bounds='clip')
                iso.fit(X_tr, Y_tr)
                p_iso_test2 = iso.predict(X_te)
                p_iso_eval = iso.predict(X_ev)
                iso_metrics_test2 = utils_calibration.calculate_all_metrics(Y_te, p_iso_test2)
                iso_metrics_eval = utils_calibration.calculate_all_metrics(Y_ev, p_iso_eval)
                
                # 3. Logistic Platt Calibration
                lr = LogisticRegression()
                lr.fit(X_tr.reshape(-1, 1), Y_tr)
                p_lr_test2 = lr.predict_proba(X_te.reshape(-1, 1))[:, 1]
                p_lr_eval = lr.predict_proba(X_ev.reshape(-1, 1))[:, 1]
                lr_metrics_test2 = utils_calibration.calculate_all_metrics(Y_te, p_lr_test2)
                lr_metrics_eval = utils_calibration.calculate_all_metrics(Y_ev, p_lr_eval)
                
                # Print to Log
                print(f"\n    === {version_name} ===")
                print(f"      [Test2 集合評估]")
                print(f"        ├─ Raw      => ECE: {raw_metrics_test2['ece']:.6f} | Brier: {raw_metrics_test2['brier']:.6f} | LogLoss: {raw_metrics_test2['logloss']:.6f}")
                print(f"        ├─ Isotonic => ECE: {iso_metrics_test2['ece']:.6f} | Brier: {iso_metrics_test2['brier']:.6f} | LogLoss: {iso_metrics_test2['logloss']:.6f}")
                print(f"        └─ Logistic => ECE: {lr_metrics_test2['ece']:.6f} | Brier: {lr_metrics_test2['brier']:.6f} | LogLoss: {lr_metrics_test2['logloss']:.6f}")
                print(f"      [Eval 外部集合評估]")
                print(f"        ├─ Raw      => ECE: {raw_metrics_eval['ece']:.6f} | Brier: {raw_metrics_eval['brier']:.6f} | LogLoss: {raw_metrics_eval['logloss']:.6f}")
                print(f"        ├─ Isotonic => ECE: {iso_metrics_eval['ece']:.6f} | Brier: {iso_metrics_eval['brier']:.6f} | LogLoss: {iso_metrics_eval['logloss']:.6f}")
                print(f"        └─ Logistic => ECE: {lr_metrics_eval['ece']:.6f} | Brier: {lr_metrics_eval['brier']:.6f} | LogLoss: {lr_metrics_eval['logloss']:.6f}")
                
                # Output detailed bin-by-bin text to log
                print(f"      [Test2 原始 Bin 分布與偏差]")
                print(print_bin_details(Y_te, X_te, bin_edges))
                print(f"      [Test2 Isotonic 校正後 Bin 分布與偏差]")
                print(print_bin_details(Y_te, p_iso_test2, bin_edges))
                print(f"      [Test2 Logistic 校正後 Bin 分布與偏差]")
                print(print_bin_details(Y_te, p_lr_test2, bin_edges))
                
                global_summaries.append({
                    'layer': layer_num,
                    'model': model_name,
                    'version': version_id,
                    'test2_raw_ece': raw_metrics_test2['ece'],
                    'test2_iso_ece': iso_metrics_test2['ece'],
                    'test2_lr_ece': lr_metrics_test2['ece'],
                    'test2_raw_brier': raw_metrics_test2['brier'],
                    'test2_iso_brier': iso_metrics_test2['brier'],
                    'test2_lr_brier': lr_metrics_test2['brier'],
                    'eval_raw_ece': raw_metrics_eval['ece'],
                    'eval_iso_ece': iso_metrics_eval['ece'],
                    'eval_lr_ece': lr_metrics_eval['ece'],
                    'eval_raw_brier': raw_metrics_eval['brier'],
                    'eval_iso_brier': iso_metrics_eval['brier'],
                    'eval_lr_brier': lr_metrics_eval['brier']
                })
                
                return {
                    'y_true_test2': Y_te, 'y_prob_raw_test2': X_te, 'y_prob_iso_test2': p_iso_test2, 'y_prob_lr_test2': p_lr_test2,
                    'y_true_eval': Y_ev, 'y_prob_raw_eval': X_ev, 'y_prob_iso_eval': p_iso_eval, 'y_prob_lr_eval': p_lr_eval
                }
                
            print(f"\n  [模型 {model_name}] 進行三種版本校正訓練...")
            
            # --- 1️⃣ 版本一：理想作弊基準組 (Whiteboard Benchmark) ---
            X_v1_test1 = p1_test1 * y1_test1 + (1.0 - p1_test1) * (1.0 - y1_test1)
            X_v1_test2 = p1_test2 * y1_test2 + (1.0 - p1_test2) * (1.0 - y1_test2)
            X_v1_eval = p1_eval * y1_eval + (1.0 - p1_eval) * (1.0 - y1_eval)
            plot_data['v1'][model_name] = run_calibration_for_pair(
                X_v1_test1, y3_test1, X_v1_test2, y3_test2, X_v1_eval, y3_eval,
                'V1', '版本一：理想作弊基準組 (Whiteboard Benchmark)'
            )
            
            # --- 2️⃣ 版本二：正統預測正確性校正組 (Standard Calibration) ---
            X_v2_test1 = np.maximum(p1_test1, 1.0 - p1_test1)
            X_v2_test2 = np.maximum(p1_test2, 1.0 - p1_test2)
            X_v2_eval = np.maximum(p1_eval, 1.0 - p1_eval)
            
            pred_y1_test1 = (p1_test1 >= 0.5).astype(int)
            pred_y1_test2 = (p1_test2 >= 0.5).astype(int)
            pred_y1_eval = (p1_eval >= 0.5).astype(int)
            
            Y_v2_test1 = (pred_y1_test1 == y1_test1).astype(int)
            Y_v2_test2 = (pred_y1_test2 == y1_test2).astype(int)
            Y_v2_eval = (pred_y1_eval == y1_eval).astype(int)
            
            plot_data['v2'][model_name] = run_calibration_for_pair(
                X_v2_test1, Y_v2_test1, X_v2_test2, Y_v2_test2, X_v2_eval, Y_v2_eval,
                'V2', '版本二：正統預測正確性校正組 (Standard Calibration)'
            )
            
            # --- 3️⃣ 版本三：改良版跨任務預測組 (Cross-Task Consistency) ---
            X_v3_test1 = np.maximum(p1_test1, 1.0 - p1_test1)
            X_v3_test2 = np.maximum(p1_test2, 1.0 - p1_test2)
            X_v3_eval = np.maximum(p1_eval, 1.0 - p1_eval)
            plot_data['v3'][model_name] = run_calibration_for_pair(
                X_v3_test1, y3_test1, X_v3_test2, y3_test2, X_v3_eval, y3_eval,
                'V3', '版本三：改良版跨任務預測組 (Cross-Task Consistency)'
            )
            
        # --- 繪製三種版本校正前後的對比圖 (3x2 網格) ---
        print(f"\n  [繪圖] 繪製第 {layer_num} 層的三種版本校正對比折線圖...")
        utils_calibration.setup_chinese_font()
        
        colors = {'SGD': '#4C72B0', 'MLP': '#55A868', 'LGB': '#C44E52', 'LR': '#8172B3', 'RF': '#CCB974'}
        markers = {'SGD': 'o', 'MLP': 's', 'LGB': '^', 'LR': 'v', 'RF': 'D'}
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 11), sharex=False, sharey=True)
        fig.suptitle(f"第 {layer_num} 層隱藏狀態 - 三種資料組版本之校正對比曲線 (Test2 測試集)", fontsize=16, fontweight='bold')
        
        versions = ['v1', 'v2', 'v3']
        version_titles = {
            'v1': '版本一：理想作弊基準組\n(Whiteboard Benchmark)',
            'v2': '版本二：正統預測正確性校正組\n(Standard Calibration)',
            'v3': '版本三：改良版跨任務預測組\n(Cross-Task Consistency)'
        }
        
        for v_idx, v_key in enumerate(versions):
            ax_iso = axes[0, v_idx]
            ax_iso.plot([0, 1], [0, 1], "k--", label="完美校正對角線", alpha=0.5)
            
            ax_lr = axes[1, v_idx]
            ax_lr.plot([0, 1], [0, 1], "k--", label="完美校正對角線", alpha=0.5)
            
            for model_name in models:
                if model_name not in plot_data[v_key]:
                    continue
                d = plot_data[v_key][model_name]
                
                # Raw vs Isotonic (Row 0)
                frac_pos_raw, mean_pred_raw, _ = utils_calibration.calculate_calibration_curve(d['y_true_test2'], d['y_prob_raw_test2'], bin_edges)
                frac_pos_iso, mean_pred_iso, _ = utils_calibration.calculate_calibration_curve(d['y_true_test2'], d['y_prob_iso_test2'], bin_edges)
                
                ece_raw = utils_calibration.calculate_ece(d['y_true_test2'], d['y_prob_raw_test2'])
                ece_iso = utils_calibration.calculate_ece(d['y_true_test2'], d['y_prob_iso_test2'])
                
                # Raw (dashed)
                ax_iso.plot(mean_pred_raw, frac_pos_raw, linestyle='--', marker=markers[model_name], markersize=4,
                            color=colors[model_name], alpha=0.4, label=f"{model_name} Raw (ECE: {ece_raw:.3f})")
                # Isotonic (solid)
                ax_iso.plot(mean_pred_iso, frac_pos_iso, linestyle='-', marker=markers[model_name], markersize=6,
                            color=colors[model_name], linewidth=2.0, label=f"{model_name} Cal (ECE: {ece_iso:.3f})")
                
                # Raw vs Logistic (Row 1)
                frac_pos_lr, mean_pred_lr, _ = utils_calibration.calculate_calibration_curve(d['y_true_test2'], d['y_prob_lr_test2'], bin_edges)
                ece_lr = utils_calibration.calculate_ece(d['y_true_test2'], d['y_prob_lr_test2'])
                
                # Raw (dashed)
                ax_lr.plot(mean_pred_raw, frac_pos_raw, linestyle='--', marker=markers[model_name], markersize=4,
                           color=colors[model_name], alpha=0.4, label=f"{model_name} Raw (ECE: {ece_raw:.3f})")
                # Logistic (solid)
                ax_lr.plot(mean_pred_lr, frac_pos_lr, linestyle='-', marker=markers[model_name], markersize=6,
                           color=colors[model_name], linewidth=2.0, label=f"{model_name} Cal (ECE: {ece_lr:.3f})")
                           
            ax_iso.set_xlim([-0.05, 1.05])
            ax_iso.set_ylim([-0.05, 1.05])
            ax_iso.set_title(f"{version_titles[v_key]}\n[Isotonic 校正]", fontsize=12, fontweight='bold')
            ax_iso.grid(True, linestyle='--', alpha=0.3)
            ax_iso.legend(loc="upper left", fontsize=7.5, framealpha=0.7)
            if v_idx == 0:
                ax_iso.set_ylabel("實際正類比例 (Actual Rate)", fontsize=11)
                
            ax_lr.set_xlim([-0.05, 1.05])
            ax_lr.set_ylim([-0.05, 1.05])
            ax_lr.set_title(f"[Logistic Platt 校正]", fontsize=12, fontweight='bold')
            ax_lr.set_xlabel("平均預測機率 / 置信度 (X 軸)", fontsize=11)
            ax_lr.grid(True, linestyle='--', alpha=0.3)
            ax_lr.legend(loc="upper left", fontsize=7.5, framealpha=0.7)
            if v_idx == 0:
                ax_lr.set_ylabel("實際正類比例 (Actual Rate)", fontsize=11)
                
        plt.tight_layout()
        fig.subplots_adjust(top=0.90)
        plot_save_path = os.path.join(base_output_dir, f"layer_{layer_num}_three_versions_calibration.png")
        fig.savefig(plot_save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    └─ 成功生成折線圖至: {plot_save_path}")
        
    df_summary = pd.DataFrame(global_summaries)
    print("\n" + "="*80)
    print(" 核心指標匯總統計表 (不用看圖也能完整掌握核心結果)")
    print("="*80)
    
    for v_id in ['V1', 'V2', 'V3']:
        v_full_names = {
            'V1': '版本一：理想作弊基準組 (Whiteboard Benchmark)',
            'V2': '版本二：正統預測正確性校正組 (Standard Calibration)',
            'V3': '版本三：改良版跨任務預測組 (Cross-Task Consistency)'
        }
        print(f"\n>>> 任務組: {v_full_names[v_id]} <<<")
        v_df = df_summary[df_summary['version'] == v_id]
        
        # Test2 ECE Table
        print("\n[Test2 集合 ECE 指標對比]")
        ece_pivot_test2 = v_df.pivot(index='model', columns='layer', values=['test2_raw_ece', 'test2_iso_ece', 'test2_lr_ece'])
        print("  * Raw ECE:")
        print(ece_pivot_test2['test2_raw_ece'].round(5).to_string())
        print("  * Isotonic Calibrated ECE:")
        print(ece_pivot_test2['test2_iso_ece'].round(5).to_string())
        print("  * Logistic Calibrated ECE:")
        print(ece_pivot_test2['test2_lr_ece'].round(5).to_string())
        
        # Test2 Brier Table
        print("\n[Test2 集合 Brier Score 指標對比]")
        brier_pivot_test2 = v_df.pivot(index='model', columns='layer', values=['test2_raw_brier', 'test2_iso_brier', 'test2_lr_brier'])
        print("  * Raw Brier:")
        print(brier_pivot_test2['test2_raw_brier'].round(5).to_string())
        print("  * Isotonic Calibrated Brier:")
        print(brier_pivot_test2['test2_iso_brier'].round(5).to_string())
        print("  * Logistic Calibrated Brier:")
        print(brier_pivot_test2['test2_lr_brier'].round(5).to_string())
        
        # Eval ECE Table
        print("\n[Eval 外部集合 ECE 指標對比]")
        ece_pivot_eval = v_df.pivot(index='model', columns='layer', values=['eval_raw_ece', 'eval_iso_ece', 'eval_lr_ece'])
        print("  * Raw ECE:")
        print(ece_pivot_eval['eval_raw_ece'].round(5).to_string())
        print("  * Isotonic Calibrated ECE:")
        print(ece_pivot_eval['eval_iso_ece'].round(5).to_string())
        print("  * Logistic Calibrated ECE:")
        print(ece_pivot_eval['eval_lr_ece'].round(5).to_string())
        
        # Eval Brier Table
        print("\n[Eval 外部集合 Brier Score 指標對比]")
        brier_pivot_eval = v_df.pivot(index='model', columns='layer', values=['eval_raw_brier', 'eval_iso_brier', 'eval_lr_brier'])
        print("  * Raw Brier:")
        print(brier_pivot_eval['eval_raw_brier'].round(5).to_string())
        print("  * Isotonic Calibrated Brier:")
        print(brier_pivot_eval['eval_iso_brier'].round(5).to_string())
        print("  * Logistic Calibrated Brier:")
        print(brier_pivot_eval['eval_lr_brier'].round(5).to_string())
        
        # Average Calibration Improvement Analysis
        avg_raw_ece = v_df['test2_raw_ece'].mean()
        avg_iso_ece = v_df['test2_iso_ece'].mean()
        avg_lr_ece = v_df['test2_lr_ece'].mean()
        print(f"\n  ├─ 總體 Test2 平均 ECE 變化: Raw ({avg_raw_ece:.4f}) -> Isotonic ({avg_iso_ece:.4f}) | Logistic ({avg_lr_ece:.4f})")
        
        avg_raw_brier = v_df['test2_raw_brier'].mean()
        avg_iso_brier = v_df['test2_iso_brier'].mean()
        avg_lr_brier = v_df['test2_lr_brier'].mean()
        print(f"  ├─ 總體 Test2 平均 Brier 變化: Raw ({avg_raw_brier:.4f}) -> Isotonic ({avg_iso_brier:.4f}) | Logistic ({avg_lr_brier:.4f})")
        
        avg_eval_raw_ece = v_df['eval_raw_ece'].mean()
        avg_eval_iso_ece = v_df['eval_iso_ece'].mean()
        avg_eval_lr_ece = v_df['eval_lr_ece'].mean()
        print(f"  ├─ 總體 Eval 平均 ECE 變化: Raw ({avg_eval_raw_ece:.4f}) -> Isotonic ({avg_eval_iso_ece:.4f}) | Logistic ({avg_eval_lr_ece:.4f})")
        
        avg_eval_raw_brier = v_df['eval_raw_brier'].mean()
        avg_eval_iso_brier = v_df['eval_iso_brier'].mean()
        avg_eval_lr_brier = v_df['eval_lr_brier'].mean()
        print(f"  └─ 總體 Eval 平均 Brier 變化: Raw ({avg_eval_raw_brier:.4f}) -> Isotonic ({avg_eval_iso_brier:.4f}) | Logistic ({avg_eval_lr_brier:.4f})")
        print("-" * 80)
        
    print("\n" + "="*80)
    print(" 所有校正模型與圖表、詳細日誌已成功輸出至 results/three_versions_calibration/ 核心目錄。")
    print("="*80)

if __name__ == "__main__":
    main()
