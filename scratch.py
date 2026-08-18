import joblib
import numpy as np

v1_cache = r"C:\Users\weiwe\Safety-training-dataset\Safety-training-dataset\cache\v1_baseline\calibration\without_pca\calibrated_predictions.pkl"
v2_cache = r"C:\Users\weiwe\Safety-training-dataset\Safety-training-dataset\cache\v2_framework\calibration\without_pca\calibrated_predictions_framework.pkl"

def analyze_v1():
    try:
        data = joblib.load(v1_cache)
        # y2 task, layer 6, eval split, MLP model
        info = data['y2'][6]['splits']['eval']['MLP']
        y1 = info['y1']
        y2 = info['y_true']
        
        mask0 = (y1 == 0)
        mask1 = (y1 == 1)
        print(f"V1: y1=0 total={np.sum(mask0)}, where y2=1 (green): {np.sum(y2[mask0] == 1)}, y2=0 (red): {np.sum(y2[mask0] == 0)}")
        print(f"V1: y1=1 total={np.sum(mask1)}, where y2=1 (green): {np.sum(y2[mask1] == 1)}, y2=0 (red): {np.sum(y2[mask1] == 0)}")
    except Exception as e:
        print("V1 error:", e)

def analyze_v2():
    try:
        data = joblib.load(v2_cache)
        info = data[6]['splits']['eval']['MLP']
        y1 = info['y1_true']
        y2 = info['y2_true']
        
        mask0 = (y1 == 0)
        mask1 = (y1 == 1)
        print(f"V2: y1=0 total={np.sum(mask0)}, where y2=1 (green): {np.sum(y2[mask0] == 1)}, y2=0 (red): {np.sum(y2[mask0] == 0)}")
        print(f"V2: y1=1 total={np.sum(mask1)}, where y2=1 (green): {np.sum(y2[mask1] == 1)}, y2=0 (red): {np.sum(y2[mask1] == 0)}")
    except Exception as e:
        print("V2 error:", e)

analyze_v1()
analyze_v2()
