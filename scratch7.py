import joblib
import numpy as np

v1_cache = r"C:\Users\weiwe\Safety-training-dataset\Safety-training-dataset\cache\v1_baseline\calibration\without_pca\calibrated_predictions.pkl"
v2_cache = r"C:\Users\weiwe\Safety-training-dataset\Safety-training-dataset\outputs\v2_framework\framework_calibration\calibration_data.joblib"

data_v1 = joblib.load(v1_cache)
info_v1 = data_v1['y2'][6]['splits']['eval']['MLP']

data_v2 = joblib.load(v2_cache)
info_v2 = data_v2['layers'][6]['YHead_MLP']['eval']

def get_hists(info, y_task_key):
    y1_mask = (info['y1'] == 0)
    pre = info['score_pre'][y1_mask]
    yt = info[y_task_key][y1_mask]
    
    green = pre[yt == 1]
    red = pre[yt == 0]
    
    bins = np.linspace(0.0, 1.0, 21)
    h_green, _ = np.histogram(green, bins=bins)
    h_red, _ = np.histogram(red, bins=bins)
    
    return h_green, h_red, len(green), len(red)

v1_green, v1_red, v1_g_total, v1_r_total = get_hists(info_v1, 'y_true')
v2_green, v2_red, v2_g_total, v2_r_total = get_hists(info_v2, 'y3')

print("--- V1 Data (MLP) ---")
print(f"Total Green (y=1): {v1_g_total}, Total Red (y=0): {v1_r_total}")
print("V1 Green bin [0.95, 1.0]:", v1_green[-1])
print("V1 Red bin [0.0, 0.05]:", v1_red[0])

print("\n--- V2 Data (YHead_MLP) ---")
print(f"Total Green (y3=1): {v2_g_total}, Total Red (y3=0): {v2_r_total}")
print("V2 Green bin [0.95, 1.0]:", v2_green[-1])
print("V2 Red bin [0.0, 0.05]:", v2_red[0])
