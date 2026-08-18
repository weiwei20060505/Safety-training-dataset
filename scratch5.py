import joblib
import numpy as np

v1_cache = r"C:\Users\weiwe\Safety-training-dataset\Safety-training-dataset\cache\v1_baseline\calibration\without_pca\calibrated_predictions.pkl"
v2_cache = r"C:\Users\weiwe\Safety-training-dataset\Safety-training-dataset\outputs\v2_framework\framework_calibration\calibration_data.joblib"

data_v1 = joblib.load(v1_cache)
info_v1 = data_v1['y2'][6]['splits']['eval']['MLP']

data_v2 = joblib.load(v2_cache)
info_v2 = data_v2['layers'][6]['MLP_Hard_Dual']['eval']

def get_hists(info, y_task_key):
    y1_mask = (info['y1'] == 0)
    pre = info['score_pre'][y1_mask]
    yt = info[y_task_key][y1_mask]
    
    green = pre[yt == 1]
    
    bins = np.linspace(0.0, 1.0, 21)
    h_green, _ = np.histogram(green, bins=bins)
    
    return h_green

v1_green = get_hists(info_v1, 'y_true')
v2_green = get_hists(info_v2, 'y3')

bins = np.linspace(0.0, 1.0, 21)
print(f"{'區間 (Confidence Bin)':<25} | {'V1 綠色筆數':<15} | {'V2 綠色筆數':<15}")
print("-" * 60)
for i in range(20):
    bin_label = f"[{bins[i]:.2f}, {bins[i+1]:.2f})" if i < 19 else f"[{bins[i]:.2f}, {bins[i+1]:.2f}]"
    print(f"{bin_label:<25} | {v1_green[i]:<15} | {v2_green[i]:<15}")

print("-" * 60)
print(f"{'總計 (Sum)':<25} | {np.sum(v1_green):<15} | {np.sum(v2_green):<15}")
