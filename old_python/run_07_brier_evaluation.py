import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.calibration import IsotonicRegression

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

def generate_metrics_line_plot(df, dataset_title, save_path):
    """
    Plots a grid comparing SGD, MLP, LGB, LR, RF across layers.
    Columns: Brier Score, Log Loss
    """
    utils_calibration.setup_chinese_font()
    models = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
    colors = {'SGD': '#4C72B0', 'MLP': '#55A868', 'LGB': '#C44E52', 'LR': '#8172B3', 'RF': '#CCB974'}
    markers = {'SGD': 'o', 'MLP': 's', 'LGB': '^', 'LR': 'v', 'RF': 'D'}
    
    metrics = ['Brier Score', 'Log Loss']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    fig.suptitle(f'指標隨層數變化趨勢 - {dataset_title}', fontsize=16, fontweight='bold', y=1.02)
    
    for c_idx, mt in enumerate(metrics):
        ax = axes[c_idx]
        df_sub = df[df['metric'] == mt]
        
        best_val = float('inf')
        best_model = None
        best_layer = None
        
        for model in models:
            df_model = df_sub[df_sub['model'] == model].sort_values('layer')
            if df_model.empty:
                continue
                
            layers = df_model['layer'].values
            vals = df_model['value'].values
            
            ax.plot(layers, vals, label=model, color=colors[model], 
                    marker=markers[model], markersize=6, linewidth=1.5, alpha=0.9)
            
            # Track best
            min_idx = np.argmin(vals)
            if vals[min_idx] < best_val:
                best_val = vals[min_idx]
                best_model = model
                best_layer = layers[min_idx]
                
        # Highlight best model
        if best_model is not None:
            ax.plot(best_layer, best_val, marker='*', color='#FFD700', 
                    markersize=14, markeredgecolor='black', markeredgewidth=1, 
                    zorder=5, label='最佳模型')
                    
        ax.set_title(f'{mt}', fontsize=14, fontweight='bold')
        ax.set_ylabel('指標值 (越低越好)', fontsize=12)
        ax.set_xlabel('特徵層數 (Layer)', fontsize=12)
            
        num_layers = int(df['layer'].max()) if not df.empty else 6
        ax.set_xticks(range(1, num_layers + 1))
        ax.grid(True, linestyle='--', alpha=0.3)
        
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=10)
        
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def main():
    base_dir = "results/safety_guardrails_evaluation"
    os.makedirs(base_dir, exist_ok=True)
    sys.stdout = DualLogger(os.path.join(base_dir, "run_07_execution_log.txt"))
    print("="*60)
    print("LLM Safety Probe - 探針校正評估腳本 (支援 y1, y2, y3 獨立任務與雙測試集交叉評估)")
    print("="*60)
    
    print("[1] 載入資料集...")
    try:
        aug_test1_dict = joblib.load("augmented_test1.pkl")
        aug_test2_dict = joblib.load("augmented_test2.pkl")
        align_test1_dict = joblib.load("aligned_test1.pkl")
        align_test2_dict = joblib.load("aligned_test2.pkl")
        
        # Load Eval Dataset
        prep_eval = DataPreprocessor("experiment_results_eval.pkl")
        prep_eval.load_data()
        X_3d_eval = prep_eval.extract_features()
        y_targets_eval = prep_eval.create_targets()
        y1_eval = y_targets_eval[0].values if hasattr(y_targets_eval[0], 'values') else y_targets_eval[0]
        y3_eval = y_targets_eval[2].values if hasattr(y_targets_eval[2], 'values') else y_targets_eval[2]
        
    except Exception as e:
        print(f"錯誤: 無法讀取資料集: {e}")
        return
        
    models_list = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
    targets_list = ['y1', 'y2', 'y3']
    
    # 用於儲存 metrics 以畫跨層折線圖
    metrics_records = {
        'data_aug': {t: {'augmented_test1': [], 'augmented_test2': [], 'aligned_test2': [], 'eval': []} for t in targets_list},
        'data_align': {t: {'aligned_test1': [], 'aligned_test2': [], 'augmented_test2': [], 'eval': []} for t in targets_list}
    }
    
    for target_name in targets_list:
        print(f"\n========================================\n處理目標任務: {target_name.upper()}\n========================================")
        
        for layer_num in range(1, 7):
            print(f"\n[{layer_num}/6] 處理第 {layer_num} 層...")
            
            # Load target-specific DataFrames
            df_aug_test1 = aug_test1_dict[target_name]
            df_aug_test2 = aug_test2_dict[target_name]
            df_align_test1 = align_test1_dict[target_name]
            df_align_test2 = align_test2_dict[target_name]
            
            X_aug_test1 = np.array(df_aug_test1['hidden_state'].tolist())[:, layer_num - 1, :]
            X_aug_test2 = np.array(df_aug_test2['hidden_state'].tolist())[:, layer_num - 1, :]
            X_align_test1 = np.array(df_align_test1['hidden_state'].tolist())[:, layer_num - 1, :]
            X_align_test2 = np.array(df_align_test2['hidden_state'].tolist())[:, layer_num - 1, :]
            X_eval = X_3d_eval[:, layer_num - 1, :]
            
            y1_aug_test1, y3_aug_test1 = df_aug_test1['y1'].values, df_aug_test1['y3'].values
            y1_aug_test2, y3_aug_test2 = df_aug_test2['y1'].values, df_aug_test2['y3'].values
            y1_align_test1, y3_align_test1 = df_align_test1['y1'].values, df_align_test1['y3'].values
            y1_align_test2, y3_align_test2 = df_align_test2['y1'].values, df_align_test2['y3'].values
            
            # 繪圖資料暫存區
            plot_data_aug = {
                'augmented_test1': {},
                'augmented_test2': {},
                'aligned_test2': {},
                'eval': {}
            }
            plot_data_align = {
                'aligned_test1': {},
                'aligned_test2': {},
                'augmented_test2': {},
                'eval': {}
            }
            
            bin_edges_dict = {m: np.linspace(0.0, 1.0, 11) for m in models_list}
            
            for model_name in models_list:
                model_path = f"results/unified_training/layer_{layer_num}/{model_name.lower()}_{target_name}_best.pkl"
                if not os.path.exists(model_path):
                    continue
                    
                clf = joblib.load(model_path)
                
                # Predict raw probabilities
                p_eval = clf.predict_proba(X_eval)[:, 1]
                p_aug_test1 = clf.predict_proba(X_aug_test1)[:, 1]
                p_aug_test2 = clf.predict_proba(X_aug_test2)[:, 1]
                p_align_test1 = clf.predict_proba(X_align_test1)[:, 1]
                p_align_test2 = clf.predict_proba(X_align_test2)[:, 1]
                
                # Feature formulation based on target_name
                if target_name == 'y1' or target_name == 'y2':
                    pre_cal_eval = np.where(y1_eval == 1, p_eval, 1.0 - p_eval)
                    pre_cal_aug_test1 = np.where(y1_aug_test1 == 1, p_aug_test1, 1.0 - p_aug_test1)
                    pre_cal_aug_test2 = np.where(y1_aug_test2 == 1, p_aug_test2, 1.0 - p_aug_test2)
                    pre_cal_align_test1 = np.where(y1_align_test1 == 1, p_align_test1, 1.0 - p_align_test1)
                    pre_cal_align_test2 = np.where(y1_align_test2 == 1, p_align_test2, 1.0 - p_align_test2)
                else: # y3
                    pre_cal_eval = np.where(y3_eval == 1, p_eval, 1.0 - p_eval)
                    pre_cal_aug_test1 = np.where(y3_aug_test1 == 1, p_aug_test1, 1.0 - p_aug_test1)
                    pre_cal_aug_test2 = np.where(y3_aug_test2 == 1, p_aug_test2, 1.0 - p_aug_test2)
                    pre_cal_align_test1 = np.where(y3_align_test1 == 1, p_align_test1, 1.0 - p_align_test1)
                    pre_cal_align_test2 = np.where(y3_align_test2 == 1, p_align_test2, 1.0 - p_align_test2)
                
                # --- data_aug model: Calibrate on augmented_test1 ---
                iso_aug = IsotonicRegression(out_of_bounds='clip')
                iso_aug.fit(pre_cal_aug_test1, y3_aug_test1)
                
                plot_data_aug['augmented_test1'][model_name] = {'y_true': y3_aug_test1, 'y_prob': iso_aug.predict(pre_cal_aug_test1), 'y1': y1_aug_test1}
                plot_data_aug['augmented_test2'][model_name] = {'y_true': y3_aug_test2, 'y_prob': iso_aug.predict(pre_cal_aug_test2), 'y1': y1_aug_test2}
                plot_data_aug['aligned_test2'][model_name] = {'y_true': y3_align_test2, 'y_prob': iso_aug.predict(pre_cal_align_test2), 'y1': y1_align_test2}
                plot_data_aug['eval'][model_name] = {'y_true': y3_eval, 'y_prob': iso_aug.predict(pre_cal_eval), 'y1': y1_eval}
                
                for split_name, pd_dict in plot_data_aug.items():
                    m = utils_calibration.calculate_all_metrics(pd_dict[model_name]['y_true'], pd_dict[model_name]['y_prob'])
                    metrics_records['data_aug'][target_name][split_name].append({'layer': layer_num, 'model': model_name, 'metric': 'Brier Score', 'value': m['brier']})
                    metrics_records['data_aug'][target_name][split_name].append({'layer': layer_num, 'model': model_name, 'metric': 'Log Loss', 'value': m['logloss']})
                    
                    # Print highly detailed logs
                    print(f"    [日誌 - Data Aug - 任務: {target_name} | 層數: {layer_num} | 評估集: {split_name} | 模型: {model_name}]")
                    print(f"      Brier Score: {m['brier']:.5f} | Log Loss: {m['logloss']:.5f} | Reliability: {m['reliability']:.5f} | Resolution: {m['resolution']:.5f} | Uncertainty: {m['uncertainty']:.5f}")
                    frac_pos, mean_pred, bin_sizes = utils_calibration.calculate_calibration_curve(pd_dict[model_name]['y_true'], pd_dict[model_name]['y_prob'], bin_edges_dict[model_name])
                    for b in range(len(mean_pred)):
                        print(f"        Bin {b+1}: 平均預測值 = {mean_pred[b]:.4f} | 實際正例率 = {frac_pos[b]:.4f} | 樣本數 = {bin_sizes[b]}")
                
                # --- data_align model: Calibrate on aligned_test1 ---
                iso_align = IsotonicRegression(out_of_bounds='clip')
                iso_align.fit(pre_cal_align_test1, y3_align_test1)
                
                plot_data_align['aligned_test1'][model_name] = {'y_true': y3_align_test1, 'y_prob': iso_align.predict(pre_cal_align_test1), 'y1': y1_align_test1}
                plot_data_align['aligned_test2'][model_name] = {'y_true': y3_align_test2, 'y_prob': iso_align.predict(pre_cal_align_test2), 'y1': y1_align_test2}
                plot_data_align['augmented_test2'][model_name] = {'y_true': y3_aug_test2, 'y_prob': iso_align.predict(pre_cal_aug_test2), 'y1': y1_aug_test2}
                plot_data_align['eval'][model_name] = {'y_true': y3_eval, 'y_prob': iso_align.predict(pre_cal_eval), 'y1': y1_eval}
                
                for split_name, pd_dict in plot_data_align.items():
                    m = utils_calibration.calculate_all_metrics(pd_dict[model_name]['y_true'], pd_dict[model_name]['y_prob'])
                    metrics_records['data_align'][target_name][split_name].append({'layer': layer_num, 'model': model_name, 'metric': 'Brier Score', 'value': m['brier']})
                    metrics_records['data_align'][target_name][split_name].append({'layer': layer_num, 'model': model_name, 'metric': 'Log Loss', 'value': m['logloss']})
                    
                    # Print highly detailed logs
                    print(f"    [日誌 - Data Align - 任務: {target_name} | 層數: {layer_num} | 評估集: {split_name} | 模型: {model_name}]")
                    print(f"      Brier Score: {m['brier']:.5f} | Log Loss: {m['logloss']:.5f} | Reliability: {m['reliability']:.5f} | Resolution: {m['resolution']:.5f} | Uncertainty: {m['uncertainty']:.5f}")
                    frac_pos, mean_pred, bin_sizes = utils_calibration.calculate_calibration_curve(pd_dict[model_name]['y_true'], pd_dict[model_name]['y_prob'], bin_edges_dict[model_name])
                      # Plot data_aug curves
            for split_name in ['augmented_test1', 'augmented_test2', 'aligned_test2', 'eval']:
                rel_path = f"results/safety_guardrails_evaluation/data_aug/02_Reliability_Curves/{split_name}/layer_{layer_num}/{target_name}_reliability.png"
                utils_calibration.plot_comparison_line(
                    plot_data_aug[split_name], bin_edges_dict,
                    f"第 {layer_num} 層可靠度曲線 (增強) - {split_name.upper()} - 任務: {target_name.upper()}", rel_path
                )
                
                rel_split_path = f"results/safety_guardrails_evaluation/data_aug/02_Reliability_Curves_split_y/{split_name}/layer_{layer_num}/{target_name}_reliability_split.png"
                utils_calibration.plot_comparison_line_split_y(
                    plot_data_aug[split_name], bin_edges_dict,
                    f"第 {layer_num} 層可靠度曲線 (增強 - 依 y_i 拆分) - {split_name.upper()} - 任務: {target_name.upper()}", rel_split_path
                )
                
                brier_bar_path = f"results/safety_guardrails_evaluation/data_aug/04_Brier_Components/{split_name}/layer_{layer_num}/{target_name}_brier_components.png"
                utils_calibration.plot_brier_components_bar(
                    plot_data_aug[split_name], bin_edges_dict,
                    f"第 {layer_num} 層分區間 Brier 分解指標柱狀圖 (增強) - {split_name.upper()} - 任務: {target_name.upper()}", brier_bar_path
                )
                
            # Plot data_align curves
            for split_name in ['aligned_test1', 'aligned_test2', 'augmented_test2', 'eval']:
                rel_path = f"results/safety_guardrails_evaluation/data_align/02_Reliability_Curves/{split_name}/layer_{layer_num}/{target_name}_reliability.png"
                utils_calibration.plot_comparison_line(
                    plot_data_align[split_name], bin_edges_dict,
                    f"第 {layer_num} 層可靠度曲線 (對齊) - {split_name.upper()} - 任務: {target_name.upper()}", rel_path
                )
                
                rel_split_path = f"results/safety_guardrails_evaluation/data_align/02_Reliability_Curves_split_y/{split_name}/layer_{layer_num}/{target_name}_reliability_split.png"
                utils_calibration.plot_comparison_line_split_y(
                    plot_data_align[split_name], bin_edges_dict,
                    f"第 {layer_num} 層可靠度曲線 (對齊 - 依 y_i 拆分) - {split_name.upper()} - 任務: {target_name.upper()}", rel_split_path
                )
                
                brier_bar_path = f"results/safety_guardrails_evaluation/data_align/04_Brier_Components/{split_name}/layer_{layer_num}/{target_name}_brier_components.png"
                utils_calibration.plot_brier_components_bar(
                    plot_data_align[split_name], bin_edges_dict,
                    f"第 {layer_num} 層分區間 Brier 分解指標柱狀圖 (對齊) - {split_name.upper()} - 任務: {target_name.upper()}", brier_bar_path
                )

    print("\n[2] 繪製全特徵層跨模型指標比較圖...")
    
    # Metrics comparison plots for data_aug
    for split_name in ['augmented_test1', 'augmented_test2', 'aligned_test2', 'eval']:
        for target_name in targets_list:
            records = metrics_records['data_aug'][target_name][split_name]
            if len(records) > 0:
                df_aug = pd.DataFrame(records)
                generate_metrics_line_plot(
                    df_aug, f"資料增強 - {split_name.upper()} - 任務: {target_name.upper()}", 
                    f"results/safety_guardrails_evaluation/data_aug/01_Metrics_Trends/{split_name}/{target_name}_metrics_trend.png"
                )
                
    # Metrics comparison plots for data_align
    for split_name in ['aligned_test1', 'aligned_test2', 'augmented_test2', 'eval']:
        for target_name in targets_list:
            records = metrics_records['data_align'][target_name][split_name]
            if len(records) > 0:
                df_align = pd.DataFrame(records)
                generate_metrics_line_plot(
                    df_align, f"資料對齊 - {split_name.upper()} - 任務: {target_name.upper()}", 
                    f"results/safety_guardrails_evaluation/data_align/01_Metrics_Trends/{split_name}/{target_name}_metrics_trend.png"
                )
    
    print("\n執行完畢！所有結果與圖表已存入 results/safety_guardrails_evaluation/ 中。")

if __name__ == '__main__':
    main()
