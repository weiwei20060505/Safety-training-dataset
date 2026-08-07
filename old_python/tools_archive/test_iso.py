import joblib
import numpy as np
from sklearn.calibration import IsotonicRegression

cache_path = r"C:\Users\weiwe\OneDrive\Desktop\Safety-training dataset\results\safety_guardrails_evaluation\cache\split\calibrated_predictions.pkl"
cache = joblib.load(cache_path)['data_align']
data = cache['y1'][6]['splits']['test1']['SGD']
y3_test1 = np.array(data['y_true'])
pre_cal_test1 = np.array(data['score_pre'])
y1_test1 = np.array(data['y1'])

mask_1_test1 = (y1_test1 == 1)
X = pre_cal_test1[mask_1_test1]
y = y3_test1[mask_1_test1]

print("X mean:", np.mean(X))
print("y mean:", np.mean(y))

iso_1 = IsotonicRegression(out_of_bounds='clip')
iso_1.fit(X, y)
pred = iso_1.predict(X)

print("pred mean:", np.mean(pred))
