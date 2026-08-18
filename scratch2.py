import joblib
import numpy as np
import os

v1_cache = r"C:\Users\weiwe\Safety-training-dataset\Safety-training-dataset\cache\v1_baseline\calibration\without_pca\calibrated_predictions.pkl"
v2_cache = r"C:\Users\weiwe\Safety-training-dataset\Safety-training-dataset\outputs\v2_framework\framework_calibration\calibration_data.joblib"

def analyze_v1():
    try:
        data = joblib.load(v1_cache)
        info = data['y2'][6]['splits']['eval']['MLP']
        y1 = info['y1']
        y2 = info['y_true']
        
        mask0 = (y1 == 0)
        mask1 = (y1 == 1)
        print("--- V1 Baseline (MLP, Layer 6, eval) ---")
        print(f"V1: y1=0 total={np.sum(mask0)}, where y2=1 (green): {np.sum(y2[mask0] == 1)}, y2=0 (red): {np.sum(y2[mask0] == 0)}")
        print(f"V1: y1=1 total={np.sum(mask1)}, where y2=1 (green): {np.sum(y2[mask1] == 1)}, y2=0 (red): {np.sum(y2[mask1] == 0)}")
    except Exception as e:
        print("V1 error:", e)

def analyze_v2():
    try:
        data = joblib.load(v2_cache)
        # Using MLP_Hard_Dual or YHead_MLP as MLP counterpart
        info = data['layers'][6]['MLP_Hard_Dual']['eval']
        y1 = info['y1']
        y3 = info['y3']  # v2 uses y3 for task instead of y2?
        
        mask0 = (y1 == 0)
        mask1 = (y1 == 1)
        print("--- V2 Framework (MLP_Hard_Dual, Layer 6, eval) ---")
        print(f"V2: y1=0 total={np.sum(mask0)}, where y3=1 (green): {np.sum(y3[mask0] == 1)}, y3=0 (red): {np.sum(y3[mask0] == 0)}")
        print(f"V2: y1=1 total={np.sum(mask1)}, where y3=1 (green): {np.sum(y3[mask1] == 1)}, y3=0 (red): {np.sum(y3[mask1] == 0)}")
    except Exception as e:
        print("V2 error:", e)

if os.path.exists(v1_cache): analyze_v1()
else: print("V1 not found")

if os.path.exists(v2_cache): analyze_v2()
else: print("V2 not found")
