import os
import sys
import numpy as np
import joblib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

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

def plot_split_density_and_delta(models_data, title, save_path):
    """
    Plots:
    1. A density (KDE) plot comparing old (model_std) and new (model_align) probabilities
       for correctness group y=0 (incorrect) and y=1 (correct).
       - Old (model_std) is drawn as a thin, light-gray, semi-transparent dashed/dotted line.
       - New (model_align) is drawn as a thick solid colored line (Blue for y=0, Orange for y=1)
         with a semi-transparent filled area.
       - Draw a vertical dashed red line at the optimal rejection threshold (intersection of y=0 and y=1 for new).
    2. A delta plot subplot showing (New Density - Old Density) for y=0 to illustrate the
       reduction of overconfidence in high confidence regions.
    """
    utils_calibration.setup_chinese_font()
    models_list = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
    
    available_models = [m for m in models_list if m in models_data]
    n_models = len(available_models)
    if n_models == 0:
        return
        
    fig, axes = plt.subplots(2, n_models, figsize=(5.5 * n_models, 9.5), sharex=True)
    if n_models == 1:
        axes = np.expand_dims(axes, axis=1) # force 2D shape (2, 1)
        
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    xs = np.linspace(0.0, 1.0, 200) # strictly evaluate within [0.0, 1.0] to prevent boundary leakage
    
    for idx, model_name in enumerate(available_models):
        ax_density = axes[0, idx]
        ax_delta = axes[1, idx]
        
        data = models_data[model_name]
        y_true = data['y_true']
        prob_std = data['y_prob_std']
        prob_align = data['y_prob_align']
        
        # Split into y=0 (incorrect) and y=1 (correct)
        p_std_0 = prob_std[y_true == 0]
        p_std_1 = prob_std[y_true == 1]
        p_align_0 = prob_align[y_true == 0]
        p_align_1 = prob_align[y_true == 1]
        
        # Helper to compute KDE density safely (fallback to flat if no variance or empty)
        def get_density(vals):
            if len(vals) < 2 or np.var(vals) < 1e-9:
                counts, bin_edges = np.histogram(vals, bins=np.linspace(0.0, 1.0, 11), density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
                return np.interp(xs, bin_centers, counts, left=0.0, right=0.0)
            try:
                kde = gaussian_kde(vals)
                return kde(xs)
            except Exception:
                counts, bin_edges = np.histogram(vals, bins=np.linspace(0.0, 1.0, 11), density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
                return np.interp(xs, bin_centers, counts, left=0.0, right=0.0)
                
        d_std_0 = get_density(p_std_0)
        d_std_1 = get_density(p_std_1)
        d_align_0 = get_density(p_align_0)
        d_align_1 = get_density(p_align_1)
        
        # ---- 1. Plot Density (Row 1) ----
        # Plot Old Model (model_std) in Background (light gray dashed lines)
        ax_density.plot(xs, d_std_0, color='#b0c4de', linestyle='--', linewidth=1.2, alpha=0.6, label='舊版 y=0 (預測錯誤)')
        ax_density.plot(xs, d_std_1, color='#ffd8b1', linestyle=':', linewidth=1.2, alpha=0.6, label='舊版 y=1 (預測正確)')
        
        # Plot New Model (model_align) in Foreground (bright colors: Blue for y=0, Orange for y=1)
        color_y0 = '#4C72B0'  # Blue
        color_y1 = '#DD8452'  # Orange
        ax_density.plot(xs, d_align_0, color=color_y0, linestyle='-', linewidth=2.5, alpha=0.95, label='新版 y=0 (預測錯誤)')
        ax_density.fill_between(xs, 0, d_align_0, color=color_y0, alpha=0.15)
        
        ax_density.plot(xs, d_align_1, color=color_y1, linestyle='-', linewidth=2.5, alpha=0.95, label='新版 y=1 (預測正確)')
        ax_density.fill_between(xs, 0, d_align_1, color=color_y1, alpha=0.15)
        
        # Find best threshold where d_align_0 and d_align_1 cross (using absolute difference argmin)
        diff_d = np.abs(d_align_0 - d_align_1)
        intersect_idx = np.argmin(diff_d)
        best_threshold = xs[intersect_idx]
        
        ax_density.axvline(best_threshold, color='#d62728', linestyle='-.', linewidth=1.5, alpha=0.8,
                           label=f'最佳切分點: {best_threshold:.2f}')
        
        ax_density.set_title(f"{model_name}", fontsize=12, fontweight='bold')
        if idx == 0:
            ax_density.set_ylabel("機率密度 (Density)", fontsize=11)
        ax_density.set_xlim([0.0, 1.0])
        ax_density.grid(True, linestyle='--', alpha=0.3)
        ax_density.legend(loc="upper center", fontsize=8.5, framealpha=0.8)
        
        # ---- 2. Plot Delta (Row 2) ----
        # Delta = New Density - Old Density for y=0 (incorrect predictions)
        delta_y0 = d_align_0 - d_std_0
        
        # Plot filled area: blue/green for negative (density decreased, representing reduction in overconfidence)
        # and red/orange for positive (density increased)
        ax_delta.fill_between(xs, 0, delta_y0, where=(delta_y0 <= 0), color='#2ca02c', alpha=0.4, label='過度自信消弭 (密度下降)')
        ax_delta.fill_between(xs, 0, delta_y0, where=(delta_y0 > 0), color='#d62728', alpha=0.4, label='密度上升')
        ax_delta.plot(xs, delta_y0, color='#333333', linewidth=1.2, alpha=0.8)
        
        # Draw a horizontal line at 0
        ax_delta.axhline(0.0, color='gray', linestyle='-', linewidth=0.8, alpha=0.7)
        
        ax_delta.set_xlabel("校正後預測機率", fontsize=11)
        if idx == 0:
            ax_delta.set_ylabel("密度差值 (New - Old)", fontsize=11)
        ax_delta.set_xlim([0.0, 1.0])
        ax_delta.grid(True, linestyle='--', alpha=0.3)
        ax_delta.legend(loc="upper left", fontsize=8.5, framealpha=0.8)
        
    plt.tight_layout()
    fig.subplots_adjust(top=0.90)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def main():
    base_output_dir = "results/correctness_split_label_plots"
    os.makedirs(base_output_dir, exist_ok=True)
    sys.stdout = DualLogger(os.path.join(base_output_dir, "split_plots_log.txt"))
    
    print("="*80)
    print(" 啟動 Step 4: y=0 / y=1 KDE 機率密度與差值改善圖繪製管線")
    print("="*80)
    
    EVAL_PATH = "experiment_results_eval.pkl"
    if not os.path.exists(EVAL_PATH):
        print(f"錯誤: 確保 {EVAL_PATH} 存在。")
        sys.exit(1)
        
    print(f"[1] 載入外部驗證集: {EVAL_PATH}")
    prep_eval = DataPreprocessor(EVAL_PATH)
    prep_eval.load_data()
    X_3d_eval = prep_eval.extract_features()
    y_targets_eval = prep_eval.create_targets()
    
    num_layers = X_3d_eval.shape[1]
    models = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
    targets = ['y1', 'y2', 'y3']
    target_names = {
        'y1': 'y1 (模型回覆安全性)',
        'y2': 'y2 (提示詞有害性)',
        'y3': 'y3 (安全判定一致性)'
    }
    
    for target_idx, target_name in enumerate(targets):
        y_eval = np.array(y_targets_eval[target_idx])
        print(f"\n處理目標任務: {target_names[target_name]}")
        
        for layer_idx in range(num_layers):
            layer_num = layer_idx + 1
            print(f"  ├─ 第 {layer_num} 層隱藏狀態")
            
            X_eval_layer = X_3d_eval[:, layer_idx, :]
            
            # Predict and collect data
            eval_data = {}
            
            for model_name in models:
                std_path = f"results/unified_training/layer_{layer_num}/{model_name.lower()}_{target_name.lower()}_calibrated.pkl"
                align_path = f"results/unified_training/layer_{layer_num}/{model_name.lower()}_{target_name.lower()}_calibrated_aligned.pkl"
                
                if not (os.path.exists(std_path) and os.path.exists(align_path)):
                    continue
                    
                cal_std = joblib.load(std_path)
                cal_align = joblib.load(align_path)
                
                probs_std = cal_std.predict_proba(X_eval_layer)[:, 1]
                probs_align = cal_align.predict_proba(X_eval_layer)[:, 1]
                
                # Load base model to get correctness labels
                base_model_path = f"results/unified_training/layer_{layer_num}/{model_name.lower()}_{target_name.lower()}_best.pkl"
                base_clf = joblib.load(base_model_path)
                
                if target_name in ['y1', 'y2']:
                    pred_eval = base_clf.predict(X_eval_layer)
                    y_eval_correct = (pred_eval == y_eval).astype(int)
                else:
                    y_eval_correct = y_eval
                
                eval_data[model_name] = {
                    'y_true': y_eval_correct,
                    'y_prob_std': probs_std,
                    'y_prob_align': probs_align
                }
                
            if not eval_data:
                continue
                
            layer_plot_dir = os.path.join(base_output_dir, f"layer_{layer_num}", target_name)
            
            # Plot KDE density plot & Delta plot
            plot_split_density_and_delta(
                eval_data,
                f"第 {layer_num} 層 - {target_names[target_name]} (機率密度 KDE 與新舊差值對比)",
                os.path.join(layer_plot_dir, "model_kde_density_and_delta_plots.png")
            )
            print(f"    └─ 成功生成對比與差值圖至 {layer_plot_dir}")
            
    print("\n" + "="*80)
    print(" Step 4: 運行完成！所有 KDE 與差值統計圖已繪製完畢。")
    print("="*80)

if __name__ == "__main__":
    main()
