import os
import sys
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from unified_train import DataPreprocessor, DataSplitter

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

def setup_chinese_font():
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'PMingLiU', 'DFKai-SB', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False 

def generate_raw_isotonic_plots(model_data_dict, dataset_key, dataset_display, target_name, layer_idx, output_folder):
    """
    完全不使用 sklearn 的 calibration_curve。
    直接提取 Isotonic Regression 產生的所有 Unique 數值作為原始 Bins 來繪圖。
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
    
    # 建立畫布
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6))
    if n_models == 1: axes = [axes]
    fig.suptitle(f'第 {layer_idx} 層 - {target_display} [{dataset_display}] (原汁原味 Isotonic 階梯)', fontsize=18, fontweight='bold')
    
    for ax, model_name in zip(axes, available_models):
        y_true = model_data_dict[model_name]['y']
        S_pred = model_data_dict[model_name]['S']
        
        # 核心邏輯：找出 Isotonic Regression 自己切出來的所有獨立階梯 (Unique Bins)
        unique_S, counts = np.unique(S_pred, return_counts=True)
        n_raw_bins = len(unique_S)
        
        # 計算每個獨立階梯裡面的實際勝率
        frac_pos = np.array([np.mean(y_true[S_pred == val]) for val in unique_S])
        
        print(f"\n  [階梯解析] {model_name} 在 {dataset_display} 總共自動切出了 {n_raw_bins} 個階梯！")
        
        # 畫完美 45 度線
        ax.plot([0, 1], [0, 1], "k--", label="完美校正線", zorder=1)
        
        # 畫出原汁原味的折線圖 (因為點很多，線條調細一點，點調小一點)
        ax.plot(unique_S, frac_pos, marker=markers.get(model_name, 'o'), markersize=4, 
                linewidth=1.2, color=colors[model_name], alpha=0.8, label=f"{model_name} (共 {n_raw_bins} 階)", zorder=2)
        
        ax.set_title(f'{model_name}', fontsize=15, fontweight='bold')
        ax.set_xlabel("校正後預測分數 (Isotonic 原始階梯點)", fontsize=12)
        if ax == axes[0]: ax.set_ylabel("實際正確比例", fontsize=12)
        else: ax.set_ylabel("") 
        
        # 固定 X 與 Y 軸範圍
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlim([-0.05, 1.05])
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc="upper left")
        
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    line_path = os.path.join(output_folder, f"raw_isotonic_lines_{target_name.lower()}_{dataset_key}.png")
    fig.savefig(line_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def process_layer_target(layer_idx, X_test, y_test, X_eval, y_eval, target_name, layer_output_dir, image_output_dir):
    models_to_plot = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
    X_test1, X_test2, y_test1, y_test2 = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
    
    y_test1_np, y_test2_np, y_eval_np = np.array(y_test1), np.array(y_test2), np.array(y_eval)
    output_folder = f"{image_output_dir}/{target_name}_png"
    os.makedirs(output_folder, exist_ok=True)
    
    dict_test1, dict_test2, dict_eval = {}, {}, {}
    
    print(f"\n===== [第 {layer_idx} 層 - 目標 {target_name}] 擷取 Isotonic 原始階梯 =====")
    
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

    generate_raw_isotonic_plots(dict_test1, 'test1', 'Test1', target_name, layer_idx, output_folder)
    generate_raw_isotonic_plots(dict_test2, 'test2', 'Test2', target_name, layer_idx, output_folder)
    generate_raw_isotonic_plots(dict_eval, 'eval', 'Eval', target_name, layer_idx, output_folder)

def main():
    base_log_dir = "results/isotonic_raw_plots"
    os.makedirs(base_log_dir, exist_ok=True)
    sys.stdout = DualLogger(os.path.join(base_log_dir, "raw_process_log.txt"))

    print("\n" + "="*80)
    print(" 啟動 Isotonic Regression 階梯透視工具")
    print("="*80)

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
        X_2d_train = X_3d_train[:, layer_idx, :]
        X_2d_eval = X_3d_eval[:, layer_idx, :]

        _, _, X_test_y1, _, _, y_test_y1, _ = DataSplitter.split_and_scale(X_2d_train, y1_train, layer_idx)
        _, _, X_test_y2, _, _, y_test_y2, _ = DataSplitter.split_and_scale(X_2d_train, y2_train, layer_idx)
        _, _, X_test_y3, _, _, y_test_y3, _ = DataSplitter.split_and_scale(X_2d_train, y3_train, layer_idx)

        layer_output_dir = f"results/unified_training/layer_{layer_idx+1}"
        image_output_dir = os.path.join(base_log_dir, f"layer_{layer_idx+1}")
        os.makedirs(image_output_dir, exist_ok=True)
        
        process_layer_target(layer_idx+1, X_test_y1, y_test_y1, X_2d_eval, y1_eval, 'y1', layer_output_dir, image_output_dir)
        process_layer_target(layer_idx+1, X_test_y2, y_test_y2, X_2d_eval, y2_eval, 'y2', layer_output_dir, image_output_dir)
        process_layer_target(layer_idx+1, X_test_y3, y_test_y3, X_2d_eval, y3_eval, 'y3', layer_output_dir, image_output_dir)

if __name__ == "__main__":
    main()