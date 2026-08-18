import joblib
import numpy as np

v1_cache = r"C:\Users\weiwe\Safety-training-dataset\Safety-training-dataset\cache\v1_baseline\calibration\without_pca\calibrated_predictions.pkl"
v2_cache = r"C:\Users\weiwe\Safety-training-dataset\Safety-training-dataset\outputs\v2_framework\framework_calibration\calibration_data.joblib"

data_v1 = joblib.load(v1_cache)
info_v1 = data_v1['y2'][6]['splits']['eval']['MLP']

data_v2 = joblib.load(v2_cache)
info_v2 = data_v2['layers'][6]['MLP_Hard_Dual']['eval']

pre_v1 = info_v1['score_pre'][info_v1['y1'] == 0]
pre_v2 = info_v2['score_pre'][info_v2['y1'] == 0]

print("V1 pre_scores (y1=0) min/max:", np.min(pre_v1), np.max(pre_v1))
print("V1 histogram counts:", np.histogram(pre_v1, bins=np.linspace(0, 1, 21))[0])

print("V2 pre_scores (y1=0) min/max:", np.min(pre_v2), np.max(pre_v2))
print("V2 histogram counts:", np.histogram(pre_v2, bins=np.linspace(0, 1, 21))[0])
