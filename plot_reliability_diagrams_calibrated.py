import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from unified_train import DataPreprocessor, DataSplitter

def get_calibrated_classifier(clf, method='isotonic'):
    """
    獲取適合當前 scikit-learn 版本的 CalibratedClassifierCV。
    相容於新版 (使用 FrozenEstimator & cv=None) 與舊版 (使用 cv='prefit') 的 scikit-learn。
    """
    try:
        from sklearn.frozen import FrozenEstimator
        return CalibratedClassifierCV(FrozenEstimator(clf), method=method, cv=None)
    except ImportError:
        return CalibratedClassifierCV(clf, method=method, cv='prefit')

def setup_chinese_font():
    """設定 Matplotlib 支援中文顯示"""
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'PMingLiU', 'DFKai-SB', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # 解決負號變方塊的問題

def plot_isotonic_regression_curve(clf, h_val, C_val, h_test, C_test):
    """
    原版單一模型 Isotonic Regression 校正曲線繪製函數
    """
    setup_chinese_font()
    
    # 1. 建立校正模型
    calibrated_clf = get_calibrated_classifier(clf, method='isotonic')
    calibrated_clf.fit(h_val, C_val) 

    # 2. 取得「校正後」在測試集上的機率分數
    S_calibrated = calibrated_clf.predict_proba(h_test)[:, 1]

    # 3. 重新計算校正後的曲線數據
    fraction_pos_cal, mean_pred_cal = calibration_curve(C_test, S_calibrated, n_bins=10, strategy='uniform')

    # 4. 畫圖驗證
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.plot(mean_pred_cal, fraction_pos_cal, "s-", color="orange", label="Isotonic Calibrated Model")
    plt.ylabel("Fraction of positives (經驗正確率 C)")
    plt.xlabel("Mean predicted value (校正後分數 S)")
    plt.ylim([-0.05, 1.05])
    plt.title('Calibration Curve after Isotonic Regression')
    plt.legend()
    try:
        plt.show()
    except Exception:
        pass
    plt.close()

def plot_all_models_calibrated(layer_idx, X_val, y_val, X_test, y_test, target_name, layer_output_dir):
    """
    計算並繪製所有模型 (SGD, MLP, LGB, LR, RF) 在進行 Isotonic Regression 校正後的 Calibration Curve 對比
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
    
    y_val_np = np.array(y_val)
    y_test_np = np.array(y_test)
    
    print(f"\n==================== [第 {layer_idx} 層 - {target_name.upper()} 已進行 Isotonic Calibration 校正] Bins 統計數據 ====================")
    
    for model_name in models_to_plot:
        model_path = os.path.join(layer_output_dir, f"{model_name.lower()}_{target_name.lower()}.pkl")
        if not os.path.exists(model_path):
            print(f"  [跳過] 找不到模型檔案: {model_path}")
            continue
            
        # 載入未校正的原模型
        clf = joblib.load(model_path)
        
        # 建立 Isotonic 校正模型，並用 validation set (驗證集) 進行 fit
        calibrated_clf = get_calibrated_classifier(clf, method='isotonic')
        calibrated_clf.fit(X_val, y_val_np)
        
        # 儲存已校正的模型，便於後續預估使用
        calibrated_model_path = os.path.join(layer_output_dir, f"{model_name.lower()}_{target_name.lower()}_calibrated.pkl")
        joblib.dump(calibrated_clf, calibrated_model_path)
        
        # 取得校正後在測試集上的機率分數 (S_calibrated)
        S_calibrated = calibrated_clf.predict_proba(X_test)[:, 1]
        
        # 計算校正後的 calibration curve 數據 (n_bins=10)
        fraction_of_positives_cal, mean_predicted_value_cal = calibration_curve(
            y_test_np, S_calibrated, n_bins=10, strategy='uniform'
        )
        
        # 輸出每個區間 of details
        print(f"\n 模型: {model_name} (已校正)")
        print(f"  {'區間 (Bin)':<22} | {'平均預測分數 (S)':<18} | {'實際正確比例 (C)':<18} | {'資料筆數':<10}")
        print("-" * 80)
        
        bin_edges = np.linspace(0, 1, 11)
        bin_indices = np.digitize(S_calibrated, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, 9)
        
        for b_idx in range(10):
            mask = bin_indices == b_idx
            count = np.sum(mask)
            mean_pred = np.mean(S_calibrated[mask]) if count > 0 else 0.0
            frac_pos = np.mean(y_test_np[mask]) if count > 0 else 0.0
            print(f"  Bin {b_idx+1:2d} ({bin_edges[b_idx]:.1f}-{bin_edges[b_idx+1]:.1f}) | {mean_pred:17.4f} | {frac_pos:17.4f} | {count:<10d}")
            
        # 繪製曲線
        plt.plot(
            mean_predicted_value_cal, 
            fraction_of_positives_cal, 
            marker=markers[model_name], 
            color=colors[model_name], 
            label=f"{model_name} (Isotonic Calibrated)",
            linewidth=1.5,
            markersize=6
        )
        
    plt.ylabel("Fraction of positives (實際正確的比例 C)")
    plt.xlabel("Mean predicted value (平均預測分數 S)")
    plt.ylim([-0.05, 1.05])
    plt.xlim([-0.05, 1.05])
    plt.title(f'Calibration Curve (Reliability Diagram) - Layer {layer_idx} {target_name.upper()} (Isotonic Calibrated)')
    plt.legend(loc="upper left")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 儲存圖片
    os.makedirs(layer_output_dir, exist_ok=True)
    save_fig_path = os.path.join(layer_output_dir, f"calibration_corrected_{target_name.lower()}.png")
    plt.savefig(save_fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[OK] 校正後對比圖表已儲存至: {save_fig_path}")

def main():
    print("\n" + "="*80)
    print(" 啟動 Isotonic Regression 校正與圖表繪製工具 (Calibrated)")
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
        print(f"[正在計算第 {layer_idx + 1} / {num_layers} 層特徵的 Isotonic Calibration]")
        print("="*60)
        
        X_2d = X_3d[:, layer_idx, :]

        # 取得驗證集與測試集資料 (驗證集用於擬合校正模型，測試集用於最終評估與畫圖)
        _, X_val_y1, X_test_y1, _, y_val_y1, y_test_y1, _ = DataSplitter.split_and_scale(X_2d, y1, layer_idx)
        _, X_val_y3, X_test_y3, _, y_val_y3, y_test_y3, _ = DataSplitter.split_and_scale(X_2d, y3, layer_idx)

        layer_output_dir = f"results/unified_training/layer_{layer_idx+1}"
        
        # 進行 Y1 (Harmful) 的 Isotonic 校正並畫圖
        plot_all_models_calibrated(layer_idx+1, X_val_y1, y_val_y1, X_test_y1, y_test_y1, 'y1', layer_output_dir)
        
        # 進行 Y3 (Consistency) 的 Isotonic 校正並畫圖
        plot_all_models_calibrated(layer_idx+1, X_val_y3, y_val_y3, X_test_y3, y_test_y3, 'y3', layer_output_dir)

    print(f"\n[OK] 所有 {num_layers} 層特徵的 Isotonic 校正與 Calibration Curve 繪製已完成！")

if __name__ == "__main__":
    main()