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
        Returns prediction correctness probabilities.
        Column 0: Probability of prediction error (incorrect)
        Column 1: Probability of prediction correctness (correct)
        """
        # # Obtain base class 1 probabilities
        # p1 = self.base_clf.predict_proba(X)[:, 1]
        
        # # Calculate probability of correctness:
        # # If p1 >= threshold, model predicts class 1, correctness probability is p1
        # # If p1 < threshold, model predicts class 0, correctness probability is 1 - p1
        # p_correct = np.where(p1 >= self.threshold, p1, 1.0 - p1)
        # p_error = 1.0 - p_correct
        
        # # Ensure dimensions match (n_samples, 2)
        # return np.column_stack([p_error, p_correct])
        return self.base_clf.predict_proba(X)
    def predict(self, X):
        """
        Predicts 1 if the correctness probability >= 0.5, else 0.
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
