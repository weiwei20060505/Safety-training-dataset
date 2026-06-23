import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from unified_train import DataPreprocessor, DataSplitter

def setup_chinese_font():
    """設定 Matplotlib 支援中文顯示"""
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'PMingLiU', 'DFKai-SB', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # 解決負號變方塊的問題

def plot_calibration_curve(clf, h_test, C_test):
    """
    原版單一模型 Calibration Curve 繪製函數
    取得模型預測的機率分數 (S_h) 並計算、繪製 Calibration Curve 與直方圖
    """
    setup_chinese_font()
    
    # 1. 取得模型預測的機率分數 (S_h)
    S_test = clf.predict_proba(h_test)[:, 1] 

    # 2. 計算校正曲線的數據 (將分數切分為 10 個 Bins)
    fraction_of_positives, mean_predicted_value = calibration_curve(C_test, S_test, n_bins=10, strategy='uniform')

    # 3. 開始畫圖
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # --- 繪製線圖 (C vs S) ---
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated (理想45度線)")
    ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="Uncalibrated Model")
    ax1.set_ylabel("Fraction of positives (經驗正確率 C)")
    ax1.set_xlabel("Mean predicted value (預測分數 S)")
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_title('Calibration Curve (Reliability Diagram)')
    ax1.legend(loc="upper left")

    # --- 繪製直方圖 ---
    ax2 = ax1.twinx()
    ax2.hist(S_test, range=(0, 1), bins=10, alpha=0.3, color='gray', label="Data Density (Histogram)")
    ax2.set_ylabel("Count (資料筆數)")
    ax2.legend(loc="lower right")

    plt.show()

def plot_all_models_calibration(layer_idx, X_test, y_test, target_name, layer_output_dir):
    """
    計算並繪製所有模型 (SGD, MLP, LGB, LR, RF) 在同一張圖上的 Calibration Curve 對比
    """
    setup_chinese_font()
    
    models_to_plot = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
    plt.figure(figsize=(10, 8))
    
    # 繪製理想 45 度線
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated (理想 45 度對角線)")
    
    colors = {
        'SGD': '#4C72B0', 
        'MLP': '#55A868', 
        'LGB': '#C44E52', 
        'LR': '#8172B3', 
        'RF': '#CCB974'
    }
    markers = {
        'SGD': 'o',
        'MLP': 's',
        'LGB': '^',
        'LR': 'v',
        'RF': 'D'
    }
    
    y_test_np = np.array(y_test)
    
    print(f"\n==================== [第 {layer_idx} 層 - {target_name.upper()} 未校正] Bins 統計數據 ====================")
    
    for model_name in models_to_plot:
        model_path = os.path.join(layer_output_dir, f"{model_name.lower()}_{target_name.lower()}.pkl")
        if not os.path.exists(model_path):
            print(f"  [跳過] 找不到模型檔案: {model_path}")
            continue
            
        clf = joblib.load(model_path)
        
        # 預測機率分數 (S)
        S_test = clf.predict_proba(X_test)[:, 1]
        
        # 計算 calibration curve 數據 (n_bins=10)
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test_np, S_test, n_bins=10, strategy='uniform'
        )
        
        # 輸出每個區間的詳細統計資訊
        print(f"\n 模型: {model_name}")
        print(f"  {'區間 (Bin)':<22} | {'平均預測分數 (S)':<18} | {'實際正確比例 (C)':<18} | {'資料筆數':<10}")
        print("-" * 80)
        
        bin_edges = np.linspace(0, 1, 11)
        bin_indices = np.digitize(S_test, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, 9)
        
        for b_idx in range(10):
            mask = bin_indices == b_idx
            count = np.sum(mask)
            mean_pred = np.mean(S_test[mask]) if count > 0 else 0.0
            frac_pos = np.mean(y_test_np[mask]) if count > 0 else 0.0
            print(f"  Bin {b_idx+1:2d} ({bin_edges[b_idx]:.1f}-{bin_edges[b_idx+1]:.1f}) | {mean_pred:17.4f} | {frac_pos:17.4f} | {count:<10d}")
            
        # 繪製曲線
        plt.plot(
            mean_predicted_value, 
            fraction_of_positives, 
            marker=markers[model_name], 
            color=colors[model_name], 
            label=f"{model_name} (Uncalibrated)",
            linewidth=1.5,
            markersize=6
        )
        
    plt.ylabel("Fraction of positives (實際正確的比例 C)")
    plt.xlabel("Mean predicted value (平均預測分數 S)")
    plt.ylim([-0.05, 1.05])
    plt.xlim([-0.05, 1.05])
    plt.title(f'Calibration Curve (Reliability Diagram) - Layer {layer_idx} {target_name.upper()} (Uncalibrated)')
    plt.legend(loc="upper left")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 儲存圖片
    os.makedirs(layer_output_dir, exist_ok=True)
    save_fig_path = os.path.join(layer_output_dir, f"calibration_uncorrected_{target_name.lower()}.png")
    plt.savefig(save_fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[OK] 未校正對比圖表已儲存至: {save_fig_path}")

def main():
    print("\n" + "="*80)
    print(" 啟動 Calibration Curve 繪製工具 (Uncalibrated)")
    print("="*80)

    # 數據檔案路徑
    DATA_PATH = "experiment_results_train_1000.pkl"
    if not os.path.exists(DATA_PATH):
        DATA_PATH = "experiment_results.pkl"
    if not os.path.exists(DATA_PATH):
        print(f"錯誤: 找不到數據檔案 {DATA_PATH}")
        sys.exit(1)

    # 讀取並預處理資料
    preprocessor = DataPreprocessor(DATA_PATH)
    df = preprocessor.load_data()
    X_3d = preprocessor.extract_features()
    y1, y3 = preprocessor.create_targets()

    num_layers = X_3d.shape[1]
    
    # 遍歷所有層進行評估與畫圖
    for layer_idx in range(num_layers):
        print(f"\n" + "="*60)
        print(f"[正在計算第 {layer_idx + 1} / {num_layers} 層特徵的 Calibration Curve]")
        print("="*60)
        
        X_2d = X_3d[:, layer_idx, :]

        # 取得測試集資料 (必須與訓練時的 DataSplitter 分割一致)
        _, _, X_test_y1, _, _, y_test_y1, _ = DataSplitter.split_and_scale(X_2d, y1, layer_idx)
        _, _, X_test_y3, _, _, y_test_y3, _ = DataSplitter.split_and_scale(X_2d, y3, layer_idx)

        layer_output_dir = f"results/unified_training/layer_{layer_idx+1}"
        
        # 繪製 Y1 (Harmful) 的 calibration curves
        plot_all_models_calibration(layer_idx+1, X_test_y1, y_test_y1, 'y1', layer_output_dir)
        
        # 繪製 Y3 (Consistency) 的 calibration curves
        plot_all_models_calibration(layer_idx+1, X_test_y3, y_test_y3, 'y3', layer_output_dir)

    print(f"\n[OK] 所有 {num_layers} 層特徵的未校正 Calibration Curve 繪製已完成！")

if __name__ == "__main__":
    main()