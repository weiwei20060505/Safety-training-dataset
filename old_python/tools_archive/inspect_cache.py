import joblib
import numpy as np
import utils_calibration

cache_path = r"C:\Users\weiwe\OneDrive\Desktop\Safety-training dataset\results\safety_guardrails_evaluation\cache\split\calibrated_predictions.pkl"
cache = joblib.load(cache_path)['data_align']
data = cache['y1'][6]['splits']['test1']['SGD']
y_true = np.array(data['y_true'])
y_prob = np.array(data['y_prob'])
y1 = np.array(data['y1'])

mask_1 = (y1 == 1)
edges = np.linspace(0.0, 1.0, 11)
frac_pos, mean_pred, _ = utils_calibration.calculate_calibration_curve(y_true[mask_1], y_prob[mask_1], edges)

print("y1=1 mask size:", np.sum(mask_1))
print("frac_pos:", frac_pos)
print("mean_pred:", mean_pred)
print("diff:", frac_pos - mean_pred)
