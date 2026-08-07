import joblib
import numpy as np

cache_path = r"C:\Users\weiwe\OneDrive\Desktop\Safety-training dataset\results\safety_guardrails_evaluation\cache\split\calibrated_predictions.pkl"
cache = joblib.load(cache_path)['data_align']
data = cache['y1'][6]['splits']['test1']['SGD']
y_true = np.array(data['y_true'])
y_prob = np.array(data['y_prob'])
y1 = np.array(data['y1'])

mask_1 = (y1 == 1)
print("y1=1 Global mean_true:", np.mean(y_true[mask_1]))
print("y1=1 Global mean_prob:", np.mean(y_prob[mask_1]))

print("y1=0 Global mean_true:", np.mean(y_true[~mask_1]))
print("y1=0 Global mean_prob:", np.mean(y_prob[~mask_1]))
