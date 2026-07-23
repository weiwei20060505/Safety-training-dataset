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

base_out = r"C:\Users\weiwe\OneDrive\Desktop\Safety-training dataset\results\safety_guardrails_evaluation\data_align\split\04_Brier_Components"

split_mapping = {
    'test1': 'aligned_test1',
    'test2': 'aligned_test2',
    'test2_cross': 'augmented_test2',
    'eval': 'eval'
}

targets = ['y1', 'y2', 'y3']
models = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
colors = {'Reliability': '#55A868', 'Resolution': '#C44E52', 'Weight': '#FFB90F'}

target_titles = {
    'y1': 'Y1 (Model Reply Safety)',
    'y2': 'Y2 (Prompt Harmfulness)',
    'y3': 'Y3 (Safety Consistency)'
}

def generate_plot(data, model_name, target_name, dataset_name, layer_num, is_dual, save_path):
    y_true = np.array(data['y_true'])
    y_prob = np.array(data['y_prob'])
    
    global_mean = np.mean(y_true) if len(y_true) > 0 else 0.0
    N = len(y_true)
    
    edges = np.linspace(0.0, 1.0, 11)
    bin_ids = np.digitize(y_prob, edges)
    
    rel_vals, res_vals, weight_vals = [], [], []
    for b in range(1, 11):
        mask = (bin_ids == b)
        if b == 10:
            mask = mask | (y_prob == 1.0)
        n_samples = np.sum(mask)
        if n_samples > 0:
            w_b = n_samples / N
            pk_bar = np.mean(y_prob[mask])
            ok_bar = np.mean(y_true[mask])
            rel_b = w_b * ((pk_bar - ok_bar) ** 2)
            res_b = w_b * ((ok_bar - global_mean) ** 2)
        else:
            w_b, rel_b, res_b = 0.0, 0.0, 0.0
        rel_vals.append(rel_b)
        res_vals.append(res_b)
        weight_vals.append(w_b)
        
    x_indices = np.arange(10)
    bin_labels = [f'{i/10:.1f}-{(i+1)/10:.1f}' for i in range(10)]
    bar_width = 0.35
    
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    metrics = utils_calibration.calculate_all_metrics(y_true, y_prob)
    title_mode = '雙 Y 軸 (Dual Axis)' if is_dual else '單 Y 軸 (Single Axis)'
    fig.suptitle(f'{model_name} - Brier 組分與 Bin 樣本佔比 ({title_mode})\n{dataset_name} | Layer {layer_num} | {target_titles[target_name]}', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    if is_dual:
        ax_rel = ax_top
        ax_res = ax_top.twinx()
        
        b1 = ax_rel.bar(x_indices - bar_width/2, rel_vals, width=bar_width, color=colors['Reliability'], label='可靠度 (Rel)', zorder=3, alpha=0.85)
        b2 = ax_res.bar(x_indices + bar_width/2, res_vals, width=bar_width, color=colors['Resolution'], label='區分度 (Res)', zorder=3, alpha=0.85)
        
        ax_rel.set_ylabel('Reliability (Rel 貢獻值)', color=colors['Reliability'], fontsize=11, fontweight='bold')
        ax_res.set_ylabel('Resolution (Res 貢獻值)', color=colors['Resolution'], fontsize=11, fontweight='bold')
        ax_rel.tick_params(axis='y', labelcolor=colors['Reliability'])
        ax_res.tick_params(axis='y', labelcolor=colors['Resolution'])
        
        # Combine legends
        lines1, labels1 = ax_rel.get_legend_handles_labels()
        lines2, labels2 = ax_res.get_legend_handles_labels()
        ax_rel.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
    else:
        ax_top.bar(x_indices - bar_width/2, rel_vals, width=bar_width, color=colors['Reliability'], label='可靠度 (Rel)', zorder=3, alpha=0.85)
        ax_top.bar(x_indices + bar_width/2, res_vals, width=bar_width, color=colors['Resolution'], label='區分度 (Res)', zorder=3, alpha=0.85)
        ax_top.set_ylabel('Rel / Res 貢獻值', fontsize=11, fontweight='bold')
        ax_top.legend(loc='upper right', fontsize=9)
        
    ax_top.grid(True, linestyle='--', alpha=0.3)
    ax_top.set_title(f"Brier: {metrics['brier']:.4f} | Rel: {metrics['reliability']:.4f} | Res: {metrics['resolution']:.4f} | Unc: {metrics['uncertainty']:.4f}", fontsize=10)
    
    # Bottom Subplot: Weight
    ax_bottom.bar(x_indices, weight_vals, width=0.5, color=colors['Weight'], edgecolor='black', linewidth=0.7, label='樣本佔比 (Weight)', zorder=3)
    ax_bottom.set_ylabel('Weight 樣本比例', fontsize=11, color='#B8860B', fontweight='bold')
    ax_bottom.set_xlabel('預測分數區間 (Bins)', fontsize=11, fontweight='bold')
    ax_bottom.set_xticks(x_indices)
    ax_bottom.set_xticklabels(bin_labels, rotation=45, fontsize=9)
    max_w = max(weight_vals) if len(weight_vals) > 0 and max(weight_vals) > 0 else 0.5
    ax_bottom.set_ylim([0, min(1.0, max_w * 1.25)])
    ax_bottom.grid(True, linestyle='--', alpha=0.3)
    ax_bottom.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()

def main():
    print("="*70)
    print("開始依據全新層級結構產出 04_Brier_Components 圖表...")
    print("結構: y1/y2/y3 -> dataset -> layer_1~6 -> dual_y / single_y -> 5 大模型圖檔")
    print("="*70)
    
    count = 0
    for target in targets:
        for layer_num in range(1, 7):
            for split_key, dataset_name in split_mapping.items():
                for model in models:
                    data = cache[target][layer_num]['splits'][split_key][model]
                    
                    for is_dual, axis_mode in [(True, 'dual_y'), (False, 'single_y')]:
                        folder = os.path.join(base_out, target, dataset_name, f'layer_{layer_num}', axis_mode)
                        save_path1 = os.path.join(folder, f'{model}_brier_components.png')
                        save_path2 = os.path.join(folder, f'{model}.png')
                        
                        generate_plot(data, model, target, dataset_name, layer_num, is_dual, save_path1)
                        generate_plot(data, model, target, dataset_name, layer_num, is_dual, save_path2)
                        count += 1
                        
    print(f"成功產出 {count} 張 Brier 組分二合一圖表！")

if __name__ == '__main__':
    main()
