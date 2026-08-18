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
    red = pre[yt == 0]
    
    h_green, _ = np.histogram(green, bins=np.linspace(0, 1, 21))
    h_red, _ = np.histogram(red, bins=np.linspace(0, 1, 21))
    
    return h_green, h_red

v1_green, v1_red = get_hists(info_v1, 'y_true')
v2_green, v2_red = get_hists(info_v2, 'y3')

print("V1 y1=0 Green (y=1) max height:", np.max(v1_green), "Total:", np.sum(v1_green))
print("V1 y1=0 Red (y=0) max height:", np.max(v1_red), "Total:", np.sum(v1_red))

print("V2 y1=0 Green (y3=1) max height:", np.max(v2_green), "Total:", np.sum(v2_green))
print("V2 y1=0 Red (y3=0) max height:", np.max(v2_red), "Total:", np.sum(v2_red))
