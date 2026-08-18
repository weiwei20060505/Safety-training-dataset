import joblib
import numpy as np

v1_cache = r"C:\Users\weiwe\Safety-training-dataset\Safety-training-dataset\cache\v1_baseline\calibration\without_pca\calibrated_predictions.pkl"
data_v1 = joblib.load(v1_cache)
info_v1 = data_v1['y2'][6]['splits']['eval']['MLP']

y1_mask = (info_v1['y1'] == 0)
yt = info_v1['y_true'][y1_mask]

pre = info_v1['score_pre'][y1_mask]
cal = info_v1['y_prob'][y1_mask]

green_pre = pre[yt == 1]
green_cal = cal[yt == 1]

bins = np.linspace(0.0, 1.0, 21)
h_green_pre, _ = np.histogram(green_pre, bins=bins)
h_green_cal, _ = np.histogram(green_cal, bins=bins)

print("Green PRE (from score_pre):")
print(h_green_pre)
print("Green CAL (from y_prob):")
print(h_green_cal)
