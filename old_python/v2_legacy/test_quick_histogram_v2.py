import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.calibration import IsotonicRegression

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation_pipeline.step3_plot_histograms import plot_quadrant_histograms
import utils_calibration

utils_calibration.setup_chinese_font()

def test_all_histograms_v2():
    parser = argparse.ArgumentParser(description="v2_20k Quick Histograms")
    parser.add_argument("--layer", type=int, default=6, help="Layer index (1 to 6)")
    parser.add_argument("--model", type=str, default="all", help="Model name (sgd, mlp, lgb, lr, rf, all)")
    args = parser.parse_args()
    
    print("="*60)
    print("v2_20k 直方圖快速測試 (Option B: 無降採樣 - 評估集 Test2)")
    print("="*60)
    
    data_dir = "data/v2_20k"
    models_dir = "models/v2_20k"
    results_dir = "results/v2_20k"
    os.makedirs(results_dir, exist_ok=True)
    
    test1_path = os.path.join(data_dir, "test1_2000.pkl")
    test2_path = os.path.join(data_dir, "test2_2000.pkl")
    
    if not (os.path.exists(test1_path) and os.path.exists(test2_path)):
        print("錯誤: 找不到 test1 或 test2 檔案，請確認 prepare_v2_20k_data.py 已完成。")
        return
        
    df_test1 = pd.read_pickle(test1_path)
    df_test2 = pd.read_pickle(test2_path)
    
    X_3d_test1 = np.array(df_test1['hidden_state'].tolist())
    X_3d_test2 = np.array(df_test2['hidden_state'].tolist())
    
    layer_num = args.layer
    print(f"測試層數: Layer {layer_num}")
    
    X_test1 = X_3d_test1[:, layer_num - 1, :]
    X_test2 = X_3d_test2[:, layer_num - 1, :]
    
    y1_test1, y2_test1, y3_test1 = df_test1['y1'].values, df_test1['y2'].values, df_test1['y3'].values
    y1_test2, y2_test2, y3_test2 = df_test2['y1'].values, df_test2['y2'].values, df_test2['y3'].values
    
    all_models = ['SGD', 'MLP', 'LGB', 'LR', 'RF']
    if args.model.lower() == 'all':
        models_to_test = all_models
    else:
        models_to_test = [args.model.upper()]
        
    for model_name in models_to_test:
        m_key = model_name.lower()
        print(f"\n========================================")
        print(f"測試模型: {model_name}")
        print(f"========================================")
        
        for target_name in ['y1', 'y2', 'y3']:
            model_path = os.path.join(models_dir, f"layer_{layer_num}", f"{m_key}_{target_name}_best.pkl")
            if not os.path.exists(model_path):
                print(f"警告: {model_path} 不存在，跳過 {model_name} - {target_name}。")
                continue
                
            clf = joblib.load(model_path)
            
            # 1. 預測 Raw Probabilities
            p_test1 = clf.predict_proba(X_test1)[:, 1]
            p_test2 = clf.predict_proba(X_test2)[:, 1]
            
            # 2. 計算 pre_cal 信心分數
            if target_name in ['y1', 'y2']:
                pre_cal_test1 = np.where(y1_test1 == 1, p_test1, 1.0 - p_test1)
                pre_cal_test2 = np.where(y1_test2 == 1, p_test2, 1.0 - p_test2)
            else:  # y3
                pre_cal_test1 = p_test1
                pre_cal_test2 = p_test2
                
            # 3. 訓練校正器 (Option B: 無降採樣，於 Test1 上訓練)
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(pre_cal_test1, y3_test1)
            
            # 4. 對 Test2 (最終評估集) 推論校正後分數
            p_cal_test2 = iso.predict(pre_cal_test2)
            
            # 5. 繪圖
            save_path = os.path.join(results_dir, f"test_quick_histogram_v2_{model_name}_{target_name}.png")
            plot_quadrant_histograms(
                pre_cal_test2, p_cal_test2, y1_test2, y2_test2, y3_test2,
                layer_num, model_name, "v2_20k Test2 (Option B)", "TEST2_EVAL", target_name, save_path
            )
            print(f"  └─ 成功！{model_name} ({target_name.upper()}) 圖片已儲存至: {save_path}")

if __name__ == "__main__":
    test_all_histograms_v2()
