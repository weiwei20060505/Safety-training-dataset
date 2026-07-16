import os
import sys
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve, CalibrationDisplay, CalibratedClassifierCV
from sklearn.model_selection import train_test_split
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

def generate_dual_calibration_plots(model_data_dict, dataset_key, dataset_display, target_name, layer_idx, output_folder):
    """
    同時生成「折線圖」與「長條圖」，並全部採用 quantile 動態等頻切分法。
    """
    setup_chinese_font()
    models = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
    colors = {'SGD': '#4C72B0', 'MLP': '#55A868', 'LGB': '#C44E52', 'LR': '#8172B3', 'RF': '#CCB974'}
    markers = {'SGD': 'o', 'MLP': 's', 'LGB': '^', 'LR': 'v', 'RF': 'D'}
    
    available_models = [m for m in models if m in model_data_dict]
    if not available_models: return

    target_display_map = {'y1': "模型回覆安全性預測", 'y2': "提示詞有害性預測", 'y3': "安全判定一致性預測"}
    target_display = target_display_map.get(target_name.lower(), target_name)
    n_models = len(available_models)
    
    # ================= 1. 繪製 1x5 折線圖 (使用 CalibrationDisplay) =================
    fig_line, axes_line = plt.subplots(1, n_models, figsize=(6 * n_models, 6))
    if n_models == 1: axes_line = [axes_line]
    fig_line.suptitle(f'第 {layer_idx} 層 - {target_display} 信賴度折線圖 [{dataset_display}] (Quantile 動態區間)', fontsize=18, fontweight='bold')
    
    for ax, model_name in zip(axes_line, available_models):
        y_true = model_data_dict[model_name]['y']
        S_pred = model_data_dict[model_name]['S']
        
        CalibrationDisplay.from_predictions(
            y_true, S_pred, n_bins=10, strategy='quantile', 
            name=model_name, color=colors[model_name], marker=markers.get(model_name, 'o'),
            linewidth=1.5, ax=ax
        )
        
        ax.set_title(f'{model_name}', fontsize=15, fontweight='bold')
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_xlabel("校正後預測分數 (S)", fontsize=12)
        if ax == axes_line[0]: ax.set_ylabel("實際正確比例", fontsize=12)
        else: ax.set_ylabel("") 
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlim([-0.05, 1.05])
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc="upper left")
        
    fig_line.tight_layout()
    fig_line.subplots_adjust(top=0.88)
    line_path = os.path.join(output_folder, f"lines_quantile_{target_name.lower()}_{dataset_key}.png")
    fig_line.savefig(line_path, dpi=150, bbox_inches='tight')
    plt.close(fig_line)

    # ================= 2. 繪製 1x5 長條圖 (計算 Quantile Bins) =================
    fig_bar, axes_bar = plt.subplots(1, n_models, figsize=(6 * n_models, 6))
    if n_models == 1: axes_bar = [axes_bar]
    fig_bar.suptitle(f'第 {layer_idx} 層 - {target_display} 信賴度長條圖 [{dataset_display}] (Quantile 動態區間)', fontsize=18, fontweight='bold')
    
    for ax, model_name in zip(axes_bar, available_models):
        y_true = model_data_dict[model_name]['y']
        S_pred = model_data_dict[model_name]['S']
        
        # 使用 sklearn 內建函數計算動態區間的真實勝率與中心點
        frac_pos, mean_pred = calibration_curve(y_true, S_pred, n_bins=10, strategy='quantile')
        
        ax.plot([0, 1], [0, 1], "k--", label="完美校正線", zorder=1)
        
        # 繪製長條圖，以動態區間的平均預測分數為 X 軸中心
        bars = ax.bar(mean_pred, frac_pos, width=0.06, color=colors[model_name], alpha=0.85, edgecolor='black', linewidth=0.7, zorder=2)
        
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')
        
        ax.set_title(f'{model_name}', fontsize=15, fontweight='bold')
        ax.set_xticks(np.linspace(0, 1, 6)) # 固定底部的刻度，避免文字重疊
        ax.set_xlabel("校正後預測分數 (S)", fontsize=12)
        if ax == axes_bar[0]: ax.set_ylabel("實際正確比例", fontsize=12)
        else: ax.set_ylabel("") 
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlim([-0.05, 1.05])
        ax.grid(True, linestyle='--', alpha=0.3)
        
    fig_bar.tight_layout()
    fig_bar.subplots_adjust(top=0.88)
    bar_path = os.path.join(output_folder, f"bars_quantile_{target_name.lower()}_{dataset_key}.png")
    fig_bar.savefig(bar_path, dpi=150, bbox_inches='tight')
    plt.close(fig_bar)


def process_layer_target(layer_idx, X_test, y_test, X_eval, y_eval, target_name, layer_output_dir, image_output_dir):
    models_to_plot = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
    X_test1, X_test2, y_test1, y_test2 = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
    
    y_test1_np, y_test2_np, y_eval_np = np.array(y_test1), np.array(y_test2), np.array(y_eval)
    output_folder = f"{image_output_dir}/{target_name}_png"
    os.makedirs(output_folder, exist_ok=True)
    
    dict_test1, dict_test2, dict_eval = {}, {}, {}
    
    print(f"\n===== [第 {layer_idx} 層 - 目標 {target_name}] Isotonic 校正與預測推論 =====")
    
    for model_name in models_to_plot:
        model_path = os.path.join(layer_output_dir, f"{model_name.lower()}_{target_name.lower()}_best.pkl")
        if not os.path.exists(model_path):
            continue
            
        clf = joblib.load(model_path)
        calibrated_clf = get_calibrated_classifier(clf, method='isotonic')
        calibrated_clf.fit(X_test1, y_test1_np)
        
        dict_test1[model_name] = {'y': y_test1_np, 'S': calibrated_clf.predict_proba(X_test1)[:, 1]}
        dict_test2[model_name] = {'y': y_test2_np, 'S': calibrated_clf.predict_proba(X_test2)[:, 1]}
        dict_eval[model_name] = {'y': y_eval_np, 'S': calibrated_clf.predict_proba(X_eval)[:, 1]}
        print(f"  └─ 模型 {model_name} 處理完成")

    # 產出圖表
    generate_dual_calibration_plots(dict_test1, 'test1', 'Test1', target_name, layer_idx, output_folder)
    generate_dual_calibration_plots(dict_test2, 'test2', 'Test2', target_name, layer_idx, output_folder)
    generate_dual_calibration_plots(dict_eval, 'eval', 'Eval', target_name, layer_idx, output_folder)

def main():
    # 建立全新且獨立的輸出資料夾
    base_log_dir = "results/isotonic_dynamic_plots"
    os.makedirs(base_log_dir, exist_ok=True)
    sys.stdout = DualLogger(os.path.join(base_log_dir, "process_log.txt"))

    print("\n" + "="*80)
    print(" 啟動 Isotonic Regression 雙圖表繪製腳本 (內建 Quantile 動態分區版)")
    print("="*80)

    # 讀取資料
    TRAIN_PATH = "experiment_results_train_10000.pkl"
    if not os.path.exists(TRAIN_PATH): TRAIN_PATH = "experiment_results_train_10000.pkl"
    EVAL_PATH = "experiment_results_eval.pkl"

    prep_train = DataPreprocessor(TRAIN_PATH)
    prep_train.load_data()
    X_3d_train = prep_train.extract_features()
    y1_train, y2_train, y3_train = prep_train.create_targets()

    prep_eval = DataPreprocessor(EVAL_PATH)
    prep_eval.load_data()
    X_3d_eval = prep_eval.extract_features()
    y1_eval, y2_eval, y3_eval = prep_eval.create_targets()

    num_layers = X_3d_train.shape[1]
    
    for layer_idx in range(num_layers):
        print(f"\n[準備計算第 {layer_idx + 1} / {num_layers} 層特徵]")
        
        X_2d_train = X_3d_train[:, layer_idx, :]
        X_2d_eval = X_3d_eval[:, layer_idx, :]

        _, _, X_test_y1, _, _, y_test_y1, _ = DataSplitter.split_and_scale(X_2d_train, y1_train, layer_idx)
        _, _, X_test_y2, _, _, y_test_y2, _ = DataSplitter.split_and_scale(X_2d_train, y2_train, layer_idx)
        _, _, X_test_y3, _, _, y_test_y3, _ = DataSplitter.split_and_scale(X_2d_train, y3_train, layer_idx)

        # 讀取舊的模型檔
        layer_output_dir = f"results/unified_training/layer_{layer_idx+1}"
        # 寫入新的圖表專屬資料夾
        image_output_dir = os.path.join(base_log_dir, f"layer_{layer_idx+1}")
        os.makedirs(image_output_dir, exist_ok=True)
        
        process_layer_target(layer_idx+1, X_test_y1, y_test_y1, X_2d_eval, y1_eval, 'y1', layer_output_dir, image_output_dir)
        process_layer_target(layer_idx+1, X_test_y2, y_test_y2, X_2d_eval, y2_eval, 'y2', layer_output_dir, image_output_dir)
        process_layer_target(layer_idx+1, X_test_y3, y_test_y3, X_2d_eval, y3_eval, 'y3', layer_output_dir, image_output_dir)

if __name__ == "__main__":
    main()