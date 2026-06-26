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

def plot_individual_calibration(model_name, S_test, y_test, target_name, layer_idx, output_dir, color):
    """
    為單一模型繪製專屬的 Calibration Curve (只保留 100 Bins 比例折線圖)
    """
    fig, ax1 = plt.subplots(figsize=(8, 6)) # 移除雙 Y 軸後，圖寬可以稍微縮小
    
    # 確保資料為 Numpy Array
    y_test_np = np.array(y_test)
    
    # 計算線圖需要的比例數據
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test_np, S_test, n_bins=100, strategy='uniform')
    
    # --- 繪製比例折線與 45 度對角線 ---
    ax1.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated (45度對角線)", zorder=1)
    ax1.plot(mean_predicted_value, fraction_of_positives, "s-", markersize=4, color=color, alpha=0.8, label=f"{model_name} (比例分布)", zorder=2)
    
    ax1.set_xlabel("Mean predicted value (預測機率分數，切分 100 區間)")
    ax1.set_ylabel("Fraction of positives (實際正確的比例)", color='black')
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_xlim([-0.02, 1.02])
    
    plt.title(f'Reliability Diagram [100 Bins] - {model_name} (Layer {layer_idx} {target_name.upper()}) [Eval Set]')
    plt.legend(loc="upper left")
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 儲存圖表
    save_path = os.path.join(output_dir, f"reliability_curve_{model_name.lower()}_{target_name.lower()}_eval.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_all_models_calibration(layer_idx, X_test, y_test, target_name, layer_output_dir):
    """
    計算並繪製所有模型的總和對比圖，並呼叫函數為每個模型畫單獨的 100 區間圖
    """
    setup_chinese_font()
    models_to_plot = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
    
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated (理想 45 度對角線)")
    
    colors = {'SGD': '#4C72B0', 'MLP': '#55A868', 'LGB': '#C44E52', 'LR': '#8172B3', 'RF': '#CCB974'}
    markers = {'SGD': 'o', 'MLP': 's', 'LGB': '^', 'LR': 'v', 'RF': 'D'}
    y_test_np = np.array(y_test)
    
    print(f"\n==================== [第 {layer_idx} 層 - {target_name.upper()} (Eval Set)] 處理中 ====================")
    output_folder = f"{layer_output_dir}/{target_name}_png"
    os.makedirs(output_folder, exist_ok=True)
    
    for model_name in models_to_plot:
        model_path = os.path.join(layer_output_dir, f"{model_name.lower()}_{target_name.lower()}_best.pkl")
        if not os.path.exists(model_path):
            print(f"  [跳過] 找不到模型檔案: {model_path}")
            continue
            
        clf = joblib.load(model_path)
        
        # 取得模型預測為 1 的機率分數 (S)
        S_test = clf.predict_proba(X_test)[:, 1]
        
        # 總對比圖畫 10 區間的大趨勢
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test_np, S_test, n_bins=10, strategy='uniform')
        plt.plot(mean_predicted_value, fraction_of_positives, marker=markers[model_name], color=colors[model_name], label=f"{model_name}", linewidth=1.5)
        
        # 呼叫函數繪製單一模型的 100 區間細緻折線圖
        plot_individual_calibration(model_name, S_test, y_test_np, target_name, layer_idx, output_folder, colors[model_name])
            
    plt.ylabel("Fraction of positives (實際正確的比例)")
    plt.xlabel("Mean predicted value (平均預測分數 S)")
    plt.ylim([-0.05, 1.05])
    plt.xlim([-0.05, 1.05])
    plt.title(f'Combined Reliability Diagram [Eval Set] - Layer {layer_idx} {target_name.upper()}')
    plt.legend(loc="upper left")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    save_fig_path = os.path.join(output_folder, f"calibration_combined_{target_name.lower()}_eval.png")
    plt.savefig(save_fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [完成] 該層圖表已儲存至: {output_folder}")

def main():
    print("\n" + "="*80)
    print(" 啟動 Reliability Diagram 繪製工具 (純比例線圖 / Eval Set 驗證)")
    print("="*80)

    # 定義檔案路徑
    EVAL_PATH = "experiment_results_eval.pkl"

    # 檢查檔案是否存在
    if not os.path.exists(EVAL_PATH):
        print(f"錯誤: 找不到評估數據檔案 {EVAL_PATH}")
        sys.exit(1)

    # 載入並處理 Eval Set (真正的測試資料)
    print(f"正在載入 Eval Set: {EVAL_PATH}...")
    eval_preprocessor = DataPreprocessor(EVAL_PATH)
    eval_df = eval_preprocessor.load_data()
    X_eval_3d = eval_preprocessor.extract_features()
    y1_eval, y3_eval = eval_preprocessor.create_targets()
    
    num_layers = X_eval_3d.shape[1]
    
    for layer_idx in range(num_layers):
        print(f"\n" + "="*60)
        print(f"[正在計算第 {layer_idx + 1} / {num_layers} 層特徵的 Calibration Curve]")
        print("="*60)
        
        # 取得 Eval 該層的 2D 特徵
        X_eval_2d = X_eval_3d[:, layer_idx, :]
        
        layer_dir = f"results/unified_training/layer_{layer_idx+1}"
        os.makedirs(layer_dir, exist_ok=True)
        
        # 繪製並驗證 Eval Set (直接傳入原始 2D 特徵，因為載入的 pipelines 內部包含各自 fitted 的 scaler)
        plot_all_models_calibration(layer_idx+1, X_eval_2d, y1_eval, 'y1', layer_dir)
        plot_all_models_calibration(layer_idx+1, X_eval_2d, y3_eval, 'y3', layer_dir)

    print(f"\n[OK] 所有 {num_layers} 層特徵的 Eval Set 校正曲線繪製已完成！")

if __name__ == "__main__":
    main()