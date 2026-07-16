import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class CorrectnessClassifierWrapper(ClassifierMixin, BaseEstimator):
    """
    A wrapper class that transforms a standard class probability classifier
    into a correctness classifier for post-hoc calibration.
    
    Instead of predicting class 0 vs class 1 probabilities, it predicts
    the probability of the base model being incorrect (class 0) vs correct (class 1).
    """
    def __init__(self, base_clf=None, threshold=0.5):
        self.base_clf = base_clf
        self.threshold = threshold
        self.classes_ = np.array([0, 1])
        self._estimator_type = "classifier"
        
    def fit(self, X, y=None):
        # The base classifier is already pre-trained (prefit)
        return self
        
    def predict_proba(self, X):
        """
        直接回傳底層基礎模型的原始分類機率。
        第 0 欄 (Column 0): 預測為類別 0 的原始機率 P(Y=0|X)
        第 1 欄 (Column 1): 預測為類別 1 的原始機率 P(Y=1|X)
        """
        return self.base_clf.predict_proba(X)
    def predict(self, X):
        """
        Predicts 1 if the correctness probability >= 0.5, else 0.
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
