import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.calibration import CalibrationDisplay, calibration_curve, CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, log_loss

# 引用你寫好的前處理模組
from unified_train import DataPreprocessor, DataSplitter

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

def get_calibrated_classifier(clf, method='isotonic'):
    try:
        from sklearn.frozen import FrozenEstimator
        return CalibratedClassifierCV(FrozenEstimator(clf), method=method, cv=None)
    except ImportError:
        return CalibratedClassifierCV(clf, method=method, cv='prefit')

def setup_chinese_font():
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'PMingLiU', 'DFKai-SB', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False 

# === 🌟 新增：計算 ECE (期望校正誤差) 的統計函式 ===
def calculate_ece(y_true, y_prob, n_bins=10):
    """
    計算 Expected Calibration Error (ECE)
    將機率分為 n_bins 等份，計算每個區間內「實際勝率」與「平均預測機率」的加權絕對誤差。
    ECE 越接近 0，代表模型機率越可靠。
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(y_prob, bin_edges) - 1
    
    ece = 0.0
    n_total = len(y_prob)
    
    for i in range(n_bins):
        # 將邊界 1.0 的預測值歸入最後一個箱子
        bin_mask = (binids == i)
        if i == n_bins - 1:
            bin_mask = bin_mask | (y_prob == 1.0)
            
        n_bin = np.sum(bin_mask)
        if n_bin > 0:
            acc = np.mean(y_true[bin_mask])
            conf = np.mean(y_prob[bin_mask])
            ece += (n_bin / n_total) * np.abs(acc - conf)
            
    return ece

# === 新增：統一格式化印出評估指標的函式 ===
def print_metrics_comparison(dataset_name, y_true, raw_prob, cal_prob):
    brier_raw = brier_score_loss(y_true, raw_prob)
    brier_cal = brier_score_loss(y_true, cal_prob)
    
    ece_raw = calculate_ece(y_true, raw_prob)
    ece_cal = calculate_ece(y_true, cal_prob)
    
    ll_raw = log_loss(y_true, raw_prob)
    ll_cal = log_loss(y_true, cal_prob)
    
    print(f"    ├─ [{dataset_name} 資料集評估]")
    print(f"    │  ▶ Brier Score : {brier_raw:.4f} ➔ {brier_cal:.4f} (越接近0越好)")
    print(f"    │  ▶ ECE (校正誤差): {ece_raw:.4f} ➔ {ece_cal:.4f} (越接近0越好)")
    print(f"    │  ▶ Log Loss    : {ll_raw:.4f} ➔ {ll_cal:.4f} (越低越好)")


def generate_calibration_plots(model_data_dict, dataset_key, dataset_display, target_name, layer_idx, output_folder):
    setup_chinese_font()
    models = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
    colors = {'SGD': '#4C72B0', 'MLP': '#55A868', 'LGB': '#C44E52', 'LR': '#8172B3', 'RF': '#CCB974'}
    markers = {'SGD': 'o', 'MLP': 's', 'LGB': '^', 'LR': 'v', 'RF': 'D'}
    
    available_models = [m for m in models if m in model_data_dict]
    if not available_models: return

    target_display_map = {'y1': "模型回覆安全性預測", 'y2': "提示詞有害性預測", 'y3': "安全判定一致性預測"}
    target_display = target_display_map.get(target_name.lower(), target_name)
    
    bin_edges = np.linspace(0, 1, 11)
    
    n_models = len(available_models)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6))
    if n_models == 1: axes = [axes]
    
    fig.suptitle(f'第 {layer_idx} 層 - {target_display} 信賴度分佈對比 [{dataset_display}] (Isotonic 校正)', fontsize=18, fontweight='bold')
    
    for ax, model_name in zip(axes, available_models):
        y_true = model_data_dict[model_name]['y']
        S_pred = model_data_dict[model_name]['S']
        
        # 繪圖功能保留
        CalibrationDisplay.from_predictions(
            y_true, S_pred, n_bins=10, strategy='uniform', name=model_name,
            color=colors[model_name], marker=markers.get(model_name, 'o'),
            linewidth=1.5, ax=ax
        )
        
        ax.set_title(f'{model_name}', fontsize=15, fontweight='bold')
        ax.set_xticks(bin_edges)
        ax.set_xlabel("校正後預測分數 (S)", fontsize=12)
        if ax == axes[0]:
            ax.set_ylabel("實際正確比例 (Fraction of positives)", fontsize=12)
        else:
            ax.set_ylabel("") 
            
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlim([-0.05, 1.05])
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc="upper left")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    bar_path = os.path.join(output_folder, f"reliability_side_by_side_calibrated_{target_name.lower()}_{dataset_key}.png")
    plt.savefig(bar_path, dpi=150, bbox_inches='tight')
    plt.close()


def process_layer_target(layer_idx, X_test, y_test, X_eval, y_eval, target_name, layer_output_dir, image_output_dir):
    models_to_plot = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
    
    X_test1, X_test2, y_test1, y_test2 = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
    
    y_test1_np, y_test2_np, y_eval_np = np.array(y_test1), np.array(y_test2), np.array(y_eval)
    output_folder = f"{image_output_dir}/{target_name}_png"
    os.makedirs(output_folder, exist_ok=True)
    
    dict_test1, dict_test2, dict_eval = {}, {}, {}
    
    target_display_map = {'y1': "模型回覆安全性預測", 'y2': "提示詞有害性預測", 'y3': "安全判定一致性預測"}
    print(f"\n===== [第 {layer_idx} 層 - {target_display_map.get(target_name.lower(), target_name)}] Isotonic 校正與指標評估 =====")
    
    for model_name in models_to_plot:
        model_path = os.path.join(layer_output_dir, f"{model_name.lower()}_{target_name.lower()}_best.pkl")
        if not os.path.exists(model_path):
            continue
            
        clf = joblib.load(model_path)
        
        # 取得「校正前 (Raw)」的機率
        prob_raw_test1 = clf.predict_proba(X_test1)[:, 1]
        prob_raw_test2 = clf.predict_proba(X_test2)[:, 1]
        prob_raw_eval  = clf.predict_proba(X_eval)[:, 1]
        
        # 訓練 Isotonic 校正器
        calibrated_clf = get_calibrated_classifier(clf, method='isotonic')
        calibrated_clf.fit(X_test1, y_test1_np)
        
        # 儲存校正後的模型
        calibrated_model_path = os.path.join(layer_output_dir, f"{model_name.lower()}_{target_name.lower()}_calibrated.pkl")
        joblib.dump(calibrated_clf, calibrated_model_path)
        
        # 取得「校正後 (Calibrated)」的機率
        prob_cal_test1 = calibrated_clf.predict_proba(X_test1)[:, 1]
        prob_cal_test2 = calibrated_clf.predict_proba(X_test2)[:, 1]
        prob_cal_eval  = calibrated_clf.predict_proba(X_eval)[:, 1]
        
        # 記錄供繪圖使用 (這裡繪製的是校正後的機率)
        dict_test1[model_name] = {'y': y_test1_np, 'S': prob_cal_test1}
        dict_test2[model_name] = {'y': y_test2_np, 'S': prob_cal_test2}
        dict_eval[model_name]  = {'y': y_eval_np,  'S': prob_cal_eval}
        
        # 印出精美的對比報表
        print(f"  [模型 {model_name} 評估報告]")
        print_metrics_comparison("Test1 (校正用)", y_test1_np, prob_raw_test1, prob_cal_test1)
        print_metrics_comparison("Test2 (未見過)", y_test2_np, prob_raw_test2, prob_cal_test2)
        print_metrics_comparison("Eval  (OOD集)", y_eval_np,  prob_raw_eval,  prob_cal_eval)
        print("    └───────────────────────────────────────────────────")

    generate_calibration_plots(dict_test1, 'test1', 'Test1 校正集', target_name, layer_idx, output_folder)
    generate_calibration_plots(dict_test2, 'test2', 'Test2 獨立測試集', target_name, layer_idx, output_folder)
    generate_calibration_plots(dict_eval, 'eval', 'Eval 外部驗證集', target_name, layer_idx, output_folder)
    print(f"  [完成] 圖表與數據已生成至: {output_folder}")

def main():
    log_dir = "results/reliability_diagrams"
    os.makedirs(log_dir, exist_ok=True)
    sys.stdout = DualLogger(os.path.join(log_dir, "calibration_metrics_log.txt"))

    print("\n" + "="*80)
    print(" 啟動 Isotonic 校正與多維度指標量測 (Brier / ECE / LogLoss)")
    print("="*80)

    TRAIN_PATH = "experiment_results_train_10000.pkl"
    if not os.path.exists(TRAIN_PATH):
        TRAIN_PATH = "experiment_results_train_10000.pkl"
    if not os.path.exists(TRAIN_PATH):
        print(f"錯誤: 找不到訓練數據檔案 {TRAIN_PATH}")
        sys.exit(1)

    print(f"[1] 載入主要數據集: {TRAIN_PATH}")
    prep_train = DataPreprocessor(TRAIN_PATH)
    prep_train.load_data()
    X_3d_train = prep_train.extract_features()
    y1_train, y2_train, y3_train = prep_train.create_targets()

    EVAL_PATH = "experiment_results_eval.pkl"
    if not os.path.exists(EVAL_PATH):
        print(f"錯誤: 找不到評估檔案 {EVAL_PATH}")
        sys.exit(1)
        
    print(f"[2] 載入外部數據集: {EVAL_PATH}")
    prep_eval = DataPreprocessor(EVAL_PATH)
    prep_eval.load_data()
    X_3d_eval = prep_eval.extract_features()
    y1_eval, y2_eval, y3_eval = prep_eval.create_targets()

    num_layers = X_3d_train.shape[1]
    
    for layer_idx in range(num_layers):
        print(f"\n" + "="*60)
        print(f"[開始分析 第 {layer_idx + 1} / {num_layers} 層隱藏狀態]")
        print("="*60)
        
        X_2d_train = X_3d_train[:, layer_idx, :]
        X_2d_eval = X_3d_eval[:, layer_idx, :]

        _, _, X_test_y1, _, _, y_test_y1, _ = DataSplitter.split_and_scale(X_2d_train, y1_train, layer_idx)
        _, _, X_test_y2, _, _, y_test_y2, _ = DataSplitter.split_and_scale(X_2d_train, y2_train, layer_idx)
        _, _, X_test_y3, _, _, y_test_y3, _ = DataSplitter.split_and_scale(X_2d_train, y3_train, layer_idx)

        layer_output_dir = f"results/unified_training/layer_{layer_idx+1}"
        image_output_dir = f"results/reliability_diagrams/layer_{layer_idx+1}"
        os.makedirs(image_output_dir, exist_ok=True)
        
        process_layer_target(layer_idx+1, X_test_y1, y_test_y1, X_2d_eval, y1_eval, 'y1', layer_output_dir, image_output_dir)
        process_layer_target(layer_idx+1, X_test_y2, y_test_y2, X_2d_eval, y2_eval, 'y2', layer_output_dir, image_output_dir)
        process_layer_target(layer_idx+1, X_test_y3, y_test_y3, X_2d_eval, y3_eval, 'y3', layer_output_dir, image_output_dir)

    print(f"\n[OK] 所有分析與校正完畢，請查看 log 檔比對 Test2 的 ECE 分數！")

if __name__ == "__main__":
    main()