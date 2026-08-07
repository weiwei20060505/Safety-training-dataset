import os
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import utils_calibration

utils_calibration.setup_chinese_font()

cache_path = r"C:\Users\weiwe\OneDrive\Desktop\Safety-training dataset\results\safety_guardrails_evaluation\cache\split\calibrated_predictions.pkl"
cache = joblib.load(cache_path)['data_align']

base_rel = r"C:\Users\weiwe\OneDrive\Desktop\Safety-training dataset\results\safety_guardrails_evaluation\data_align\split\02_Reliability_Curves"
base_split = r"C:\Users\weiwe\OneDrive\Desktop\Safety-training dataset\results\safety_guardrails_evaluation\data_align\split\02_Reliability_Curves_split_y"

targets = ['y1', 'y2', 'y3']
models_list = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
colors = {'SGD': '#4C72B0', 'MLP': '#55A868', 'LGB': '#C44E52', 'LR': '#8172B3', 'RF': '#CCB974'}
markers = {'SGD': 'o', 'MLP': 's', 'LGB': '^', 'LR': 'v', 'RF': 'D'}

target_display_names = {
    'y1': 'Y1 (模型回應安全性)',
    'y2': 'Y2 (提示詞有害性)',
    'y3': 'Y3 (安全判定一致性)'
}

split_display_names = {
    'test1': 'Aligned Test 1',
    'test2': 'Aligned Test 2',
    'eval': 'Eval (外部評估集)'
}

splits_list = ['test1', 'test2', 'eval']

# ==========================================
# 1. 繪製標準 Reliability Curves (3x6 Grid)
# ==========================================
def plot_standard_reliability_cell(ax, target, layer_num, split_key, bin_method):
    ax.plot([0, 1], [0, 1], "k--", label="完美校正線", alpha=0.6)
    
    for model_name in models_list:
        data_test1 = cache[target][layer_num]['splits']['test1'][model_name]
        data_curr = cache[target][layer_num]['splits'][split_key][model_name]
        
        y_true = np.array(data_curr['y_true'])
        y_prob = np.array(data_curr['y_prob'])
        
        # Derive edges from test1 calibrated predictions
        p_train = np.array(data_test1['y_prob'])
        if bin_method == 'native':
            edges = utils_calibration.get_native_bins(p_train)
        else: # adaptive
            edges = utils_calibration.get_adaptive_bins(p_train, n_bins=10)
            
        frac_pos, mean_pred, _ = utils_calibration.calculate_calibration_curve(y_true, y_prob, edges)
        m = utils_calibration.calculate_all_metrics(y_true, y_prob)
        
        color = colors.get(model_name, '#333333')
        marker = markers.get(model_name, 'o')
        ax.plot(mean_pred, frac_pos, marker=marker, color=color, linewidth=1.5,
                label=f"{model_name} (Brier: {m['brier']:.4f})")
                
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc="upper left", fontsize=6, framealpha=0.8)

def generate_standard_3x6(bin_method):
    out_dir = os.path.join(base_rel, bin_method)
    os.makedirs(out_dir, exist_ok=True)
    method_title = '原生區間邊界 (Native Bins)' if bin_method == 'native' else '自適應合併區間 (Adaptive Bins)'
    
    for target in targets:
        fig, axes = plt.subplots(3, 6, figsize=(24, 12), sharex=True, sharey=True)
        fig.suptitle(f'標準 Reliability Diagrams 3x6 矩陣圖 [{method_title}]\n任務: {target_display_names[target]} | 資料對齊組', 
                     fontsize=18, fontweight='bold', y=0.98)
        
        for r_idx, s_key in enumerate(splits_list):
            for c_idx, layer_num in enumerate(range(1, 7)):
                ax = axes[r_idx, c_idx]
                plot_standard_reliability_cell(ax, target, layer_num, s_key, bin_method)
                
                cell_title = f"{split_display_names[s_key]} - L{layer_num}"
                ax.set_title(cell_title, fontsize=11, fontweight='bold', pad=4)
                
                if c_idx == 0:
                    ax.set_ylabel(f"{split_display_names[s_key]}\n實際正確比例", fontsize=10, fontweight='bold')
                if r_idx == 2:
                    ax.set_xlabel("平均預測機率 (S)", fontsize=10, fontweight='bold')
                    
        plt.tight_layout()
        fig.subplots_adjust(top=0.91)
        
        out_path = os.path.join(out_dir, f'{target}_reliability_{bin_method}_3x6_grid.png')
        plt.savefig(out_path, dpi=160, bbox_inches='tight')
        plt.close()
        print(f"Generated standard 3x6 grid [{bin_method}] for {target} at: {out_path}")

# ==========================================
# 2. 繪製 Split y1 Reliability Curves (3x6 Grid - Target Centric & Model Centric)
# ==========================================
def plot_split_y_cell(ax, data_curr, data_test1, model_name, bin_method):
    y_true = np.array(data_curr['y_true'])
    y_prob = np.array(data_curr['y_prob'])
    y1 = np.array(data_curr['y1'])
    
    y_prob_train = np.array(data_test1['y_prob'])
    y1_train = np.array(data_test1['y1'])
    
    ax.plot([0, 1], [0, 1], "k--", label="完美校正線", alpha=0.5)
    
    mask_1 = (y1 == 1)
    mask_0 = (y1 == 0)
    mask_1_train = (y1_train == 1)
    mask_0_train = (y1_train == 0)
    
    color = colors.get(model_name, '#333333')
    marker = markers.get(model_name, 'o')
    
    # y1 == 1
    if np.sum(mask_1) > 0:
        p_tr1 = y_prob_train[mask_1_train] if np.sum(mask_1_train) > 0 else y_prob_train
        edges_1 = utils_calibration.get_native_bins(p_tr1) if bin_method == 'native' else utils_calibration.get_adaptive_bins(p_tr1, n_bins=10)
        frac_pos_1, mean_pred_1, _ = utils_calibration.calculate_calibration_curve(y_true[mask_1], y_prob[mask_1], edges_1)
        m1 = utils_calibration.calculate_all_metrics(y_true[mask_1], y_prob[mask_1])
        ax.plot(mean_pred_1, frac_pos_1, marker=marker, color=color, linestyle='-',
                label=f"y1=1 (Brier: {m1['brier']:.4f})", linewidth=1.5)
                
    # y1 == 0
    if np.sum(mask_0) > 0:
        p_tr0 = y_prob_train[mask_0_train] if np.sum(mask_0_train) > 0 else y_prob_train
        edges_0 = utils_calibration.get_native_bins(p_tr0) if bin_method == 'native' else utils_calibration.get_adaptive_bins(p_tr0, n_bins=10)
        frac_pos_0, mean_pred_0, _ = utils_calibration.calculate_calibration_curve(y_true[mask_0], y_prob[mask_0], edges_0)
        m0 = utils_calibration.calculate_all_metrics(y_true[mask_0], y_prob[mask_0])
        ax.plot(mean_pred_0, frac_pos_0, marker=marker, color=color, linestyle='--',
                label=f"y1=0 (Brier: {m0['brier']:.4f})", linewidth=1.5, alpha=0.8)
                
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.8)

# Target-centric 3x6 for split_y (rows=datasets, cols=layers)
def generate_split_y_target_3x6(bin_method):
    out_dir = os.path.join(base_split, bin_method)
    os.makedirs(out_dir, exist_ok=True)
    method_title = '原生區間邊界 (Native Bins)' if bin_method == 'native' else '自適應合併區間 (Adaptive Bins)'
    
    for target in targets:
        fig, axes = plt.subplots(3, 6, figsize=(24, 12), sharex=True, sharey=True)
        fig.suptitle(f'Split y1 Reliability Diagrams 3x6 矩陣圖 [{method_title}]\n任務: {target_display_names[target]} | 資料對齊組', 
                     fontsize=18, fontweight='bold', y=0.98)
        
        for r_idx, s_key in enumerate(splits_list):
            for c_idx, layer_num in enumerate(range(1, 7)):
                ax = axes[r_idx, c_idx]
                # Combine models in plot
                ax.plot([0, 1], [0, 1], "k--", label="完美校正線", alpha=0.5)
                for model_name in models_list:
                    data_curr = cache[target][layer_num]['splits'][s_key][model_name]
                    data_test1 = cache[target][layer_num]['splits']['test1'][model_name]
                    plot_split_y_cell(ax, data_curr, data_test1, model_name, bin_method)
                    
                cell_title = f"{split_display_names[s_key]} - L{layer_num}"
                ax.set_title(cell_title, fontsize=11, fontweight='bold', pad=4)
                
                if c_idx == 0:
                    ax.set_ylabel(f"{split_display_names[s_key]}\n實際正確比例", fontsize=10, fontweight='bold')
                if r_idx == 2:
                    ax.set_xlabel("平均預測機率 (S)", fontsize=10, fontweight='bold')
                    
        plt.tight_layout()
        fig.subplots_adjust(top=0.91)
        
        out_path = os.path.join(out_dir, f'{target}_reliability_split_{bin_method}_3x6_grid.png')
        plt.savefig(out_path, dpi=160, bbox_inches='tight')
        plt.close()
        print(f"Generated split_y target 3x6 grid [{bin_method}] for {target} at: {out_path}")

# Model-centric 3x6 for split_y (rows=y1..y3, cols=layers) for Aligned Test 2
def generate_split_y_model_3x6(bin_method):
    out_dir = os.path.join(base_split, bin_method)
    os.makedirs(out_dir, exist_ok=True)
    method_title = '原生區間邊界 (Native Bins)' if bin_method == 'native' else '自適應合併區間 (Adaptive Bins)'
    
    for model_name in models_list:
        fig, axes = plt.subplots(3, 6, figsize=(24, 12), sharex=True, sharey=True)
        fig.suptitle(f'單一模型 Split y1 Reliability Diagrams 3x6 矩陣圖 [{method_title}]\n模型: {model_name} | 資料集: Aligned Test 2', 
                     fontsize=18, fontweight='bold', y=0.98)
        
        for r_idx, target in enumerate(targets):
            for c_idx, layer_num in enumerate(range(1, 7)):
                ax = axes[r_idx, c_idx]
                data_curr = cache[target][layer_num]['splits']['test2'][model_name]
                data_test1 = cache[target][layer_num]['splits']['test1'][model_name]
                plot_split_y_cell(ax, data_curr, data_test1, model_name, bin_method)
                
                cell_title = f"{target.upper()} - L{layer_num}"
                ax.set_title(cell_title, fontsize=11, fontweight='bold', pad=4)
                
                if c_idx == 0:
                    ax.set_ylabel(f"{target_display_names[target]}\n實際正確比例", fontsize=10, fontweight='bold')
                if r_idx == 2:
                    ax.set_xlabel("平均預測機率 (S)", fontsize=10, fontweight='bold')
                    
        plt.tight_layout()
        fig.subplots_adjust(top=0.91)
        
        out_path = os.path.join(out_dir, f'{model_name}_aligned_test2_{bin_method}_3x6_grid.png')
        plt.savefig(out_path, dpi=160, bbox_inches='tight')
        plt.close()
        print(f"Generated split_y model 3x6 grid [{bin_method}] for {model_name} at: {out_path}")

def main():
    print("="*70)
    print("開始產出 Native Bins 與 Adaptive Bins 兩種方法的 3x6 矩陣圖...")
    print("="*70)
    
    for bin_method in ['native', 'adaptive']:
        print(f"\n>>> 處理方法: {bin_method.upper()} <<<")
        generate_standard_3x6(bin_method)
        generate_split_y_target_3x6(bin_method)
        generate_split_y_model_3x6(bin_method)
        
    print("\n所有 Native 與 Adaptive 3x6 矩陣圖全數繪製完成！")

if __name__ == '__main__':
    main()
