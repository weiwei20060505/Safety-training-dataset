import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
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
            safe_msg = message.encode(encoding, errors='replace').decode(encoding)
            self.terminal.write(safe_msg)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def setup_chinese_font():
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'PMingLiU', 'DFKai-SB', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False 

def generate_calibration_plots(model_data_dict, dataset_key, dataset_display, target_name, layer_idx, output_folder):
    setup_chinese_font()
    models = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
    colors = {'SGD': '#4C72B0', 'MLP': '#55A868', 'LGB': '#C44E52', 'LR': '#8172B3', 'RF': '#CCB974'}
    markers = {'SGD': 'o', 'MLP': 's', 'LGB': '^', 'LR': 'v', 'RF': 'D'}
    
    available_models = [m for m in models if m in model_data_dict]
    if not available_models: return

    target_display_map = {'y1': "模型回覆安全性預測", 'y2': "提示詞有害性預測", 'y3': "安全判定一致性預測"}
    target_display = target_display_map.get(target_name.lower(), target_name)
    
    # 1. 繪製折線圖
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], "k--", label="完美校正線 (理想 45 度對角線)")
    
    for model_name in available_models:
        y_true = model_data_dict[model_name]['y']
        S_pred = model_data_dict[model_name]['S']
        
        frac_pos, mean_pred = calibration_curve(y_true, S_pred, n_bins=10, strategy='uniform')
        plt.plot(mean_pred, frac_pos, marker=markers[model_name], color=colors[model_name], label=f"{model_name}", linewidth=1.5)
            
    bin_edges = np.linspace(0, 1, 11)
    plt.xticks(bin_edges)
    plt.ylabel("實際正確的比例 (Fraction of positives)")
    plt.xlabel("平均預測機率值 (未校正預測分數 S)")
    plt.ylim([-0.05, 1.05])
    plt.xlim([-0.05, 1.05])
    plt.title(f'信賴度對比折線圖 [{dataset_display}] - 第 {layer_idx} 層 - {target_display} (未校正)')
    plt.legend(loc="upper left")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    line_path = os.path.join(output_folder, f"calibration_lines_uncorrected_{target_name.lower()}_{dataset_key}.png")
    plt.savefig(line_path, dpi=150, bbox_inches='tight')
    plt.close()

    # 2. 繪製 1x5 並排長條圖
    n_models = len(available_models)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6))
    if n_models == 1: axes = [axes]
    
    fig.suptitle(f'第 {layer_idx} 層 - {target_display} 信賴度分佈對比 [{dataset_display}] (未校正)', fontsize=18, fontweight='bold')
    
    for ax, model_name in zip(axes, available_models):
        y_true = model_data_dict[model_name]['y']
        S_pred = model_data_dict[model_name]['S']
        
        frac_pos, mean_pred = calibration_curve(y_true, S_pred, n_bins=10, strategy='uniform')
        
        ax.plot([0, 1], [0, 1], "k--", label="完美校正線", zorder=1)
        bars = ax.bar(mean_pred, frac_pos, width=0.08, color=colors[model_name], alpha=0.85, edgecolor='black', linewidth=0.7, zorder=2)
        
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), 
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')
        
        ax.set_title(f'{model_name}', fontsize=15, fontweight='bold')
        ax.set_xticks(bin_edges)
        ax.set_xlabel("未校正預測分數 (S)", fontsize=12)
        if ax == axes[0]:
            ax.set_ylabel("實際正確比例 (Fraction of positives)", fontsize=12)
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlim([-0.05, 1.05])
        ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    bar_path = os.path.join(output_folder, f"reliability_side_by_side_uncorrected_{target_name.lower()}_{dataset_key}.png")
    plt.savefig(bar_path, dpi=150, bbox_inches='tight')
    plt.close()

def process_layer_target(layer_idx, X_test, y_test, X_eval, y_eval, target_name, layer_output_dir, image_output_dir):
    models_to_plot = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
    
    X_test1, X_test2, y_test1, y_test2 = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
    
    y_test1_np = np.array(y_test1)
    y_test2_np = np.array(y_test2)
    y_eval_np = np.array(y_eval)
    
    output_folder = f"{image_output_dir}/{target_name}_png"
    os.makedirs(output_folder, exist_ok=True)
    
    dict_test1 = {}
    dict_test2 = {}
    dict_eval = {}
    
    target_display_map = {'y1': "模型回覆安全性預測", 'y2': "提示詞有害性預測", 'y3': "安全判定一致性預測"}
    print(f"\n===== [第 {layer_idx} 層 - {target_display_map.get(target_name.lower(), target_name)}] 處理未校正預測 =====")
    
    for model_name in models_to_plot:
        model_path = os.path.join(layer_output_dir, f"{model_name.lower()}_{target_name.lower()}_best.pkl")
        if not os.path.exists(model_path):
            print(f"  [跳過] 找不到模型檔案: {model_path}")
            continue
            
        clf = joblib.load(model_path)
        
        dict_test1[model_name] = {'y': y_test1_np, 'S': clf.predict_proba(X_test1)[:, 1]}
        dict_test2[model_name] = {'y': y_test2_np, 'S': clf.predict_proba(X_test2)[:, 1]}
        dict_eval[model_name] = {'y': y_eval_np, 'S': clf.predict_proba(X_eval)[:, 1]}
        print(f"  └─ 模型 {model_name} 推論完成")

    generate_calibration_plots(dict_test1, 'test1', 'Test1 基準對照集', target_name, layer_idx, output_folder)
    generate_calibration_plots(dict_test2, 'test2', 'Test2 測試集', target_name, layer_idx, output_folder)
    generate_calibration_plots(dict_eval, 'eval', 'Eval 外部驗證集', target_name, layer_idx, output_folder)
    print(f"  [完成] Test1, Test2, Eval 圖表已儲存至: {output_folder}")

def main():
    log_dir = "results/reliability_diagrams"
    os.makedirs(log_dir, exist_ok=True)
    sys.stdout = DualLogger(os.path.join(log_dir, "uncorrected_log.txt"))

    print("\n" + "="*80)
    print(" 啟動未校正 Reliability Diagram 繪製工具 (動態分段對齊版)")
    print("="*80)

    TRAIN_PATH = "experiment_results_train.pkl"
    if not os.path.exists(TRAIN_PATH):
        TRAIN_PATH = "experiment_results_train_10000.pkl"
    if not os.path.exists(TRAIN_PATH):
        print(f"錯誤: 找不到訓練數據檔案 {TRAIN_PATH}")
        sys.exit(1)

    print(f"[1] 正在載入主要數據集: {TRAIN_PATH}...")
    prep_train = DataPreprocessor(TRAIN_PATH)
    prep_train.load_data()
    X_3d_train = prep_train.extract_features()
    y1_train, y2_train, y3_train = prep_train.create_targets()

    EVAL_PATH = "experiment_results_eval.pkl"
    if not os.path.exists(EVAL_PATH):
        print(f"錯誤: 找不到評估數據檔案 {EVAL_PATH}，請確保檔案存在。")
        sys.exit(1)
        
    print(f"[2] 正在載入外部評估數據集: {EVAL_PATH}...")
    prep_eval = DataPreprocessor(EVAL_PATH)
    prep_eval.load_data()
    X_3d_eval = prep_eval.extract_features()
    y1_eval, y2_eval, y3_eval = prep_eval.create_targets()

    num_layers = X_3d_train.shape[1]
    
    for layer_idx in range(num_layers):
        print(f"\n" + "="*60)
        print(f"[準備計算第 {layer_idx + 1} / {num_layers} 層特徵]")
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

    print(f"\n[OK] 所有 {num_layers} 層特徵的未校正圖表繪製已全部完成！")

if __name__ == "__main__":
    main()