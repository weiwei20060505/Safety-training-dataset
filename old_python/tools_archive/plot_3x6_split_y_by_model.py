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

base_out = r"C:\Users\weiwe\OneDrive\Desktop\Safety-training dataset\results\safety_guardrails_evaluation\data_align\split\02_Reliability_Curves_split_y"
os.makedirs(base_out, exist_ok=True)

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

split_mapping = {
    'test1': 'aligned_test1',
    'test2': 'aligned_test2',
    'eval': 'eval'
}

edges = np.linspace(0.0, 1.0, 11)

def plot_cell_split_y(ax, data, model_name):
    y_true = np.array(data['y_true'])
    y_prob = np.array(data['y_prob'])
    y1 = np.array(data['y1'])
    
    ax.plot([0, 1], [0, 1], "k--", label="完美校正線", alpha=0.5)
    
    mask_1 = (y1 == 1)
    mask_0 = (y1 == 0)
    
    color = colors.get(model_name, '#333333')
    marker = markers.get(model_name, 'o')
    
    # Plot y1 == 1
    if np.sum(mask_1) > 0:
        frac_pos_1, mean_pred_1, _ = utils_calibration.calculate_calibration_curve(y_true[mask_1], y_prob[mask_1], edges)
        m1 = utils_calibration.calculate_all_metrics(y_true[mask_1], y_prob[mask_1])
        ax.plot(mean_pred_1, frac_pos_1, marker=marker, color=color, linestyle='-',
                label=f"y1=1 (Brier: {m1['brier']:.4f})", linewidth=1.5)
                
    # Plot y1 == 0
    if np.sum(mask_0) > 0:
        frac_pos_0, mean_pred_0, _ = utils_calibration.calculate_calibration_curve(y_true[mask_0], y_prob[mask_0], edges)
        m0 = utils_calibration.calculate_all_metrics(y_true[mask_0], y_prob[mask_0])
        ax.plot(mean_pred_0, frac_pos_0, marker=marker, color=color, linestyle='--',
                label=f"y1=0 (Brier: {m0['brier']:.4f})", linewidth=1.5, alpha=0.8)
                
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.8)

# 1. Generate 5 images (one per model): 3 Rows (y1, y2, y3) x 6 Cols (L1~L6) for Aligned Test 2
def generate_model_3x6_targets(model_name):
    fig, axes = plt.subplots(3, 6, figsize=(24, 12), sharex=True, sharey=True)
    fig.suptitle(f'單一模型跨層級與任務之 Reliability Diagrams (Split y1) - 模型: {model_name}\n(資料集: Aligned Test 2 | 3x6 矩陣圖)', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    rows = ['y1', 'y2', 'y3']
    for r_idx, task in enumerate(rows):
        for c_idx, layer_num in enumerate(range(1, 7)):
            ax = axes[r_idx, c_idx]
            data = cache[task][layer_num]['splits']['test2'][model_name]
            plot_cell_split_y(ax, data, model_name)
            
            cell_title = f"{task.upper()} - L{layer_num}"
            ax.set_title(cell_title, fontsize=11, fontweight='bold', pad=4)
            
            if c_idx == 0:
                ax.set_ylabel(f"{target_display_names[task]}\n實際正確比例", fontsize=10, fontweight='bold')
            if r_idx == 2:
                ax.set_xlabel("平均預測機率 (S)", fontsize=10, fontweight='bold')
                
    plt.tight_layout()
    fig.subplots_adjust(top=0.91)
    
    out1 = os.path.join(base_out, f'{model_name}_aligned_test2_3x6_grid.png')
    plt.savefig(out1, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"Generated model 3x6 target grid for {model_name} at: {out1}")

# 2. Generate 5 images per target: 3 Rows (test1, test2, eval) x 6 Cols (L1~L6) for each model
def generate_model_3x6_datasets(model_name, task):
    fig, axes = plt.subplots(3, 6, figsize=(24, 12), sharex=True, sharey=True)
    fig.suptitle(f'單一模型跨層級與資料集之 Reliability Diagrams (Split y1) - 模型: {model_name} | 任務: {target_display_names[task]}\n(3x6 矩陣圖)', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    splits = ['test1', 'test2', 'eval']
    for r_idx, s_key in enumerate(splits):
        for c_idx, layer_num in enumerate(range(1, 7)):
            ax = axes[r_idx, c_idx]
            data = cache[task][layer_num]['splits'][s_key][model_name]
            plot_cell_split_y(ax, data, model_name)
            
            cell_title = f"{split_display_names[s_key]} - L{layer_num}"
            ax.set_title(cell_title, fontsize=11, fontweight='bold', pad=4)
            
            if c_idx == 0:
                ax.set_ylabel(f"{split_display_names[s_key]}\n實際正確比例", fontsize=10, fontweight='bold')
            if r_idx == 2:
                ax.set_xlabel("平均預測機率 (S)", fontsize=10, fontweight='bold')
                
    plt.tight_layout()
    fig.subplots_adjust(top=0.91)
    
    out_dir = os.path.join(base_out, task)
    os.makedirs(out_dir, exist_ok=True)
    out2 = os.path.join(out_dir, f'{model_name}_3x6_datasets_grid.png')
    plt.savefig(out2, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"Generated model 3x6 dataset grid for {model_name} ({task}) at: {out2}")

def main():
    print("="*70)
    print("開始產出 02_Reliability_Curves_split_y 分模型 3x6 綜合矩陣圖...")
    print("="*70)
    
    for m in models_list:
        generate_model_3x6_targets(m)
        for t in ['y1', 'y2', 'y3']:
            generate_model_3x6_datasets(m, t)
            
    print("所有分模型 3x6 矩陣圖產出完畢！")

if __name__ == '__main__':
    main()
