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

from evaluation_pipeline.step3_plot_histograms import plot_quadrant_histograms
import utils_calibration

utils_calibration.setup_chinese_font()

def main():
    base_dir = "results/v2_20k/02_Safety_Evaluation"
    modes = ["baseline", "split"]
    
    print("="*60)
    print("v2_20k 評估與校正 Pipeline - 步驟三: 四象限直方圖產出")
    print("="*60)
    
    for mode in modes:
        print(f"\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n繪製校正模式直方圖: {mode.upper()}\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        
        preds_path = os.path.join(base_dir, "cache", mode, "calibrated_predictions.pkl")
        if not os.path.exists(preds_path):
            print(f"警告: 模式 {mode.upper()} 的快取檔案不存在，跳過。")
            continue
            
        predictions_cache = joblib.load(preds_path)
        targets_list = ['y1', 'y2', 'y3']
        models_list = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
        
        hist_base_dir = os.path.join(base_dir, "03_Quadrant_Histograms", mode)
        
        for eval_set in ['test1', 'test2']:
            for layer_num in range(1, 7):
                save_layer_dir = os.path.join(hist_base_dir, eval_set, f"layer_{layer_num}")
                os.makedirs(save_layer_dir, exist_ok=True)
                
                for target_name in targets_list:
                    cache_layer = predictions_cache[target_name].get(layer_num)
                    if not cache_layer: continue
                    
                    models_data = cache_layer['splits'][eval_set]
                    
                    for model_name in models_list:
                        data = models_data.get(model_name)
                        if not data: continue
                        
                        y1 = data['y1']
                        y2 = data['y2']
                        y3 = data['y3']
                        post_scores = data['y_prob']
                        score_pre = data['score_pre']
                        
                        save_path = os.path.join(save_layer_dir, f"{target_name}_{model_name}_histogram.png")
                        dataset_title = f"v2_20k ({mode.upper()})"
                        
                        plot_quadrant_histograms(
                            score_pre, post_scores, y1, y2, y3,
                            layer_num, model_name, dataset_title, eval_set.upper(), target_name, save_path
                        )
                print(f"  └─ Layer {layer_num} ({eval_set.upper()}) 直方圖繪製完成！")
                
    print("\n所有直方圖繪製完畢！")

if __name__ == '__main__':
    main()
