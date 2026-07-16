import os
import sys
import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV

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

def get_calibrated_classifier(clf, method='isotonic'):
    try:
        from sklearn.frozen import FrozenEstimator
        return CalibratedClassifierCV(FrozenEstimator(clf), method=method, cv=None)
    except ImportError:
        return CalibratedClassifierCV(clf, method=method, cv='prefit')

def main():
    base_output_dir = "results/correctness_reliability_plots"
    os.makedirs(base_output_dir, exist_ok=True)
    sys.stdout = DualLogger(os.path.join(base_output_dir, "baseline_process_log.txt"))
    
    print("="*80)
    print(" 啟動 Step 1: 基礎模型校正與評估繪圖管線 (升級命名與目錄版)")
    print("="*80)
    
    # Load dataset
    TRAIN_PATH = "experiment_results_train_10000.pkl"
    EVAL_PATH = "experiment_results_eval.pkl"
    
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(EVAL_PATH):
        print(f"錯誤: 確保 {TRAIN_PATH} 與 {EVAL_PATH} 存在。")
        sys.exit(1)
        
    print(f"[1] 載入基準 10000 訓練集: {TRAIN_PATH}")
    prep_train = DataPreprocessor(TRAIN_PATH)
    prep_train.load_data()
    X_3d_train = prep_train.extract_features()
    y_targets_train = prep_train.create_targets() # y1, y2, y3
    
    print(f"[2] 載入外部評估驗證集: {EVAL_PATH}")
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
    
    for target_idx, target_name in enumerate(targets):
        y_train = y_targets_train[target_idx]
        y_eval = y_targets_eval[target_idx]
        
        print("\n" + "="*70)
        print(f"開始處理目標任務: {target_names[target_name]}")
        print("="*70)
        
        for layer_idx in range(num_layers):
            layer_num = layer_idx + 1
            print(f"\n--- [第 {layer_num} / {num_layers} 層隱藏狀態] ---")
            
            X_2d_train = X_3d_train[:, layer_idx, :]
            X_2d_eval = X_3d_eval[:, layer_idx, :]
            
            # Split datasets
            _, _, X_test, _, _, y_test, _ = DataSplitter.split_and_scale(X_2d_train, y_train, layer_idx)
            X_test1, X_test2, y_test1, y_test2 = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
            
            y_test1_np = np.array(y_test1)
            y_test2_np = np.array(y_test2)
            y_eval_np = np.array(y_eval)
            
            # Setup output path
            target_plot_dir = os.path.join(base_output_dir, f"layer_{layer_num}", target_name)
            os.makedirs(target_plot_dir, exist_ok=True)
            
            # Dictionaries to hold model prediction data
            raw_data = {}      # uncalibrated predictions
            cal_data = {}      # calibrated predictions
            
            native_edges = {}
            adaptive_edges = {}
            uniform_edges = {}
            
            for model_name in models:
                model_path = f"results/unified_training/layer_{layer_num}/{model_name.lower()}_{target_name.lower()}_best.pkl"
                if not os.path.exists(model_path):
                    continue
                    
                clf = joblib.load(model_path)
                
                # Generate correctness targets
                if target_name in ['y1', 'y2']:
                    pred_test1 = clf.predict(X_test1)
                    pred_test2 = clf.predict(X_test2)
                    pred_eval = clf.predict(X_2d_eval)
                    
                    y_test1_cal_target = (pred_test1 == y_test1_np).astype(int)
                    y_test2_cal_target = (pred_test2 == y_test2_np).astype(int)
                    y_eval_cal_target = (pred_eval == y_eval_np).astype(int)
                    
                    clf_wrapped = CorrectnessClassifierWrapper(clf, threshold=0.5)
                else:
                    y_test1_cal_target = y_test1_np
                    y_test2_cal_target = y_test2_np
                    y_eval_cal_target = y_eval_np
                    
                    clf_wrapped = clf
                
                # 1. Uncalibrated Predictions (predict_proba)
                prob_raw_test1 = clf_wrapped.predict_proba(X_test1)[:, 1]
                prob_raw_test2 = clf_wrapped.predict_proba(X_test2)[:, 1]
                prob_raw_eval = clf_wrapped.predict_proba(X_2d_eval)[:, 1]
                
                raw_data[model_name] = {
                    'test1': {'y_true': y_test1_cal_target, 'y_prob': prob_raw_test1},
                    'test2': {'y_true': y_test2_cal_target, 'y_prob': prob_raw_test2},
                    'eval': {'y_true': y_eval_cal_target, 'y_prob': prob_raw_eval}
                }
                
                # 2. Fit Isotonic Calibration on test1
                calibrated_clf = get_calibrated_classifier(clf_wrapped, method='isotonic')
                calibrated_clf.fit(X_test1, y_test1_cal_target)
                
                # Save calibrated model (Requirement 2)
                calibrated_model_path = f"results/unified_training/layer_{layer_num}/{model_name.lower()}_{target_name.lower()}_calibrated.pkl"
                joblib.dump(calibrated_clf, calibrated_model_path)
                
                # 3. Calibrated Predictions
                prob_cal_test1 = calibrated_clf.predict_proba(X_test1)[:, 1]
                prob_cal_test2 = calibrated_clf.predict_proba(X_test2)[:, 1]
                prob_cal_eval = calibrated_clf.predict_proba(X_2d_eval)[:, 1]
                
                cal_data[model_name] = {
                    'test1': {'y_true': y_test1_cal_target, 'y_prob': prob_cal_test1},
                    'test2': {'y_true': y_test2_cal_target, 'y_prob': prob_cal_test2},
                    'eval': {'y_true': y_eval_cal_target, 'y_prob': prob_cal_eval}
                }
                
                # 4. Determine Bin Edges using Calibrated predictions on test1
                native_edges[model_name] = utils_calibration.get_native_bins(prob_cal_test1)
                adaptive_edges[model_name] = utils_calibration.get_adaptive_bins(prob_cal_test1, n_bins=10)
                uniform_edges[model_name] = np.linspace(0.0, 1.0, 11)
                
                print(f"  [模型 {model_name}] Isotonic 擬合完成：")
                print(f"    ├─ 原汁原味階梯數 (Native Bins): {len(np.unique(prob_cal_test1))}")
                print(f"    └─ 自定義動態箱數 (Adaptive Bins): {len(adaptive_edges[model_name]) - 1}")
                
                # Print Metrics to Log
                raw_metrics = utils_calibration.calculate_all_metrics(y_test2_cal_target, prob_raw_test2)
                cal_metrics = utils_calibration.calculate_all_metrics(y_test2_cal_target, prob_cal_test2)
                print(f"    ├─ Test2 ECE 變化: {raw_metrics['ece']:.4f} ➔ {cal_metrics['ece']:.4f}")
                print(f"    └─ Test2 Brier 變化: {raw_metrics['brier']:.4f} ➔ {cal_metrics['brier']:.4f}")
            
            # --- 1. 繪製未校正折線圖 (model_uncal - Requirement 1) ---
            print("  [繪圖] 繪製未校正 (Baseline) 信賴度對比折線圖...")
            raw_plot_data_test1 = {m: raw_data[m]['test1'] for m in raw_data}
            raw_plot_data_test2 = {m: raw_data[m]['test2'] for m in raw_data}
            raw_plot_data_eval = {m: raw_data[m]['eval'] for m in raw_data}
            
            uncal_dir = os.path.join(target_plot_dir, "model_uncal", "uniform")
            utils_calibration.plot_comparison_line(
                raw_plot_data_test1, uniform_edges, 
                f"第 {layer_num} 層 - {target_names[target_name]} (未校正 - Test1)",
                os.path.join(uncal_dir, f"model_uncal_on_data_std_test1_bin_uniform_lines.png")
            )
            utils_calibration.plot_comparison_line(
                raw_plot_data_test2, uniform_edges, 
                f"第 {layer_num} 層 - {target_names[target_name]} (未校正 - Test2)",
                os.path.join(uncal_dir, f"model_uncal_on_data_std_test2_bin_uniform_lines.png")
            )
            utils_calibration.plot_comparison_line(
                raw_plot_data_eval, uniform_edges, 
                f"第 {layer_num} 層 - {target_names[target_name]} (未校正 - Eval)",
                os.path.join(uncal_dir, f"model_uncal_on_data_eval_bin_uniform_lines.png")
            )
            
            # --- 2. 繪製標準校正模型圖表 (model_std - Requirement 3 & 4) ---
            print("  [繪圖] 繪製標準校正模型對比圖表...")
            cal_plot_data_test1 = {m: cal_data[m]['test1'] for m in cal_data}
            cal_plot_data_test2 = {m: cal_data[m]['test2'] for m in cal_data}
            cal_plot_data_eval = {m: cal_data[m]['eval'] for m in cal_data}
            
            std_native_dir = os.path.join(target_plot_dir, "model_std", "native")
            std_adaptive_dir = os.path.join(target_plot_dir, "model_std", "adaptive")
            
            # --- Native Binning ---
            # test1 (完美 45 度對角線)
            utils_calibration.plot_comparison_line(
                cal_plot_data_test1, native_edges,
                f"第 {layer_num} 層 - {target_names[target_name]} (model_std on data_std_test1 - 內建區間)",
                os.path.join(std_native_dir, "model_std_on_data_std_test1_bin_native_lines.png")
            )
            utils_calibration.plot_side_by_side_bars(
                cal_plot_data_test1, native_edges,
                f"第 {layer_num} 層 - {target_names[target_name]} (model_std on data_std_test1 - 內建區間)",
                os.path.join(std_native_dir, "model_std_on_data_std_test1_bin_native_bars.png")
            )
            # test2
            utils_calibration.plot_comparison_line(
                cal_plot_data_test2, native_edges,
                f"第 {layer_num} 層 - {target_names[target_name]} (model_std on data_std_test2 - 內建區間)",
                os.path.join(std_native_dir, "model_std_on_data_std_test2_bin_native_lines.png")
            )
            utils_calibration.plot_side_by_side_bars(
                cal_plot_data_test2, native_edges,
                f"第 {layer_num} 層 - {target_names[target_name]} (model_std on data_std_test2 - 內建區間)",
                os.path.join(std_native_dir, "model_std_on_data_std_test2_bin_native_bars.png")
            )
            # eval
            utils_calibration.plot_comparison_line(
                cal_plot_data_eval, native_edges,
                f"第 {layer_num} 層 - {target_names[target_name]} (model_std on data_eval - 內建區間)",
                os.path.join(std_native_dir, "model_std_on_data_eval_bin_native_lines.png")
            )
            utils_calibration.plot_side_by_side_bars(
                cal_plot_data_eval, native_edges,
                f"第 {layer_num} 層 - {target_names[target_name]} (model_std on data_eval - 內建區間)",
                os.path.join(std_native_dir, "model_std_on_data_eval_bin_native_bars.png")
            )
            
            # --- Adaptive Binning ---
            # test1 (完美 45 度對角線)
            utils_calibration.plot_comparison_line(
                cal_plot_data_test1, adaptive_edges,
                f"第 {layer_num} 層 - {target_names[target_name]} (model_std on data_std_test1 - 動態區間)",
                os.path.join(std_adaptive_dir, "model_std_on_data_std_test1_bin_adaptive_lines.png")
            )
            utils_calibration.plot_side_by_side_bars(
                cal_plot_data_test1, adaptive_edges,
                f"第 {layer_num} 層 - {target_names[target_name]} (model_std on data_std_test1 - 動態區間)",
                os.path.join(std_adaptive_dir, "model_std_on_data_std_test1_bin_adaptive_bars.png")
            )
            # test2
            utils_calibration.plot_comparison_line(
                cal_plot_data_test2, adaptive_edges,
                f"第 {layer_num} 層 - {target_names[target_name]} (model_std on data_std_test2 - 動態區間)",
                os.path.join(std_adaptive_dir, "model_std_on_data_std_test2_bin_adaptive_lines.png")
            )
            utils_calibration.plot_side_by_side_bars(
                cal_plot_data_test2, adaptive_edges,
                f"第 {layer_num} 層 - {target_names[target_name]} (model_std on data_std_test2 - 動態區間)",
                os.path.join(std_adaptive_dir, "model_std_on_data_std_test2_bin_adaptive_bars.png")
            )
            # eval
            utils_calibration.plot_comparison_line(
                cal_plot_data_eval, adaptive_edges,
                f"第 {layer_num} 層 - {target_names[target_name]} (model_std on data_eval - 動態區間)",
                os.path.join(std_adaptive_dir, "model_std_on_data_eval_bin_adaptive_lines.png")
            )
            utils_calibration.plot_side_by_side_bars(
                cal_plot_data_eval, adaptive_edges,
                f"第 {layer_num} 層 - {target_names[target_name]} (model_std on data_eval - 動態區間)",
                os.path.join(std_adaptive_dir, "model_std_on_data_eval_bin_adaptive_bars.png")
            )
            
    print("\n" + "="*80)
    print(" Step 1: 運行完成！所有基準圖表與校正模型已保存至新結構。")
    print("="*80)

if __name__ == "__main__":
    main()
