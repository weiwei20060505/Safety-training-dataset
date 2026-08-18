import joblib
import numpy as np

v1_cache_pca = r"C:\Users\weiwe\Safety-training-dataset\Safety-training-dataset\cache\v1_baseline\calibration\with_pca\calibrated_predictions.pkl"

data_v1 = joblib.load(v1_cache_pca)

def get_hists(info, y_task_key):
    y1_mask = (info['y1'] == 0)
    pre = info['score_pre'][y1_mask]
    yt = info[y_task_key][y1_mask]
    
    green = pre[yt == 1]
    
    bins = np.linspace(0.0, 1.0, 21)
    h_green, _ = np.histogram(green, bins=bins)
    return h_green

print("--- V1 Data (MLP, with PCA) ---")
info_v1 = data_v1['y2'][6]['splits']['eval']['MLP']
green = get_hists(info_v1, 'y_true')
print(f"Split eval: Total Green (y=1): {np.sum(green)}, Green bin [0.95, 1.0]: {green[-1]}")
