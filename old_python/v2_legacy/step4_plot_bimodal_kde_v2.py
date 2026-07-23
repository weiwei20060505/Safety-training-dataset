import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import utils_calibration

utils_calibration.setup_chinese_font()

class DualLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding="utf-8")
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

def main():
    base_dir = "results/v2_20k/02_Safety_Evaluation"
    modes = ["baseline", "split"]
    
    log_file = os.path.join(base_dir, "execution_log.txt")
    if os.path.exists(os.path.dirname(log_file)):
        sys.stdout = DualLogger(log_file)
        
    print("\n" + "="*70)
    print("v2_20k 評估與校正 Pipeline - 步驟四: 真實標籤分群 KDE/Histogram 雙峰圖與極詳細 Log 產出")
    print("="*70)
    
    targets_list = ['y1', 'y2', 'y3']
    models_list = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
    
    for mode in modes:
        print(f"\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n繪製與分析雙峰分佈模式: {mode.upper()}\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        
        preds_path = os.path.join(base_dir, "cache", mode, "calibrated_predictions.pkl")
        if not os.path.exists(preds_path):
            print(f"警告: 模式 {mode.upper()} 的快取檔案不存在，跳過。")
            continue
            
        predictions_cache = joblib.load(preds_path)
        bimodal_base_dir = os.path.join(base_dir, "05_Bimodal_KDE_Histograms", mode)
        
        for eval_set in ['test1', 'test2']:
            for layer_num in range(1, 7):
                save_layer_dir = os.path.join(bimodal_base_dir, eval_set, f"layer_{layer_num}")
                os.makedirs(save_layer_dir, exist_ok=True)
                
                for target_name in targets_list:
                    cache_layer = predictions_cache[target_name].get(layer_num)
                    if not cache_layer: continue
                    
                    models_data = cache_layer['splits'][eval_set]
                    
                    for model_name in models_list:
                        data = models_data.get(model_name)
                        if not data: continue
                        
                        y_true = data['y_true']
                        post_scores = data['y_prob']
                        score_pre = data['score_pre']
                        
                        header_title = f"Mode: {mode.upper()} | EvalSet: {eval_set.upper()} | Layer: {layer_num} | Model: {model_name} | Target: {target_name.upper()}"
                        
                        # 1. 印出極詳細可讀之文字日誌 (不看圖即可瞭解數據與指標)
                        utils_calibration.print_detailed_bimodal_log(header_title, y_true, score_pre, post_scores)
                        
                        # 2. 繪製並儲存 KDE/Histogram 雙峰圖
                        save_path = os.path.join(save_layer_dir, f"{target_name}_{model_name}_bimodal_kde.png")
                        plot_title = f"v2_20k 預測機率雙峰分佈 (True Label 0 vs 1)\n({header_title})"
                        
                        utils_calibration.plot_bimodal_kde_histogram(
                            y_true, score_pre, post_scores, plot_title, save_path
                        )
                        
                print(f"  └─ Layer {layer_num} ({eval_set.upper()}) 雙峰 KDE/直方圖繪製與 Log 紀錄完成！")
                
    print("\n步驟四：所有雙峰 KDE/直方圖繪製與詳細數值日誌寫入完畢！")

if __name__ == '__main__':
    main()
