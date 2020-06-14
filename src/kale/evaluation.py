import netcal.metrics
import numpy as np
from sklearn.metrics import accuracy_score


class EvalStats:
    """
    Class for computing evaluation statistics of classifiers, including calibration metrics

    :param y_true: integer array of shape (n_samples,)
    :param predicted_proba: array of shape (n_samples, n_classes)
    """
    def __init__(self, y_true: np.ndarray, predicted_proba: np.ndarray):
        assert len(y_true.shape) == 1, f"y_true has to be 1-dimensional, instead got shape: {y_true.shape}"
        assert len(predicted_proba.shape) == 2 and predicted_proba.shape[0] == len(y_true), \
            f"predicted_probabilities have to be of shape (#samples, #classes), instead got {predicted_proba.shape}"
        self.n_samples = len(y_true)
        self.y_true = y_true
        self.y_pred = predicted_proba.argmax(axis=1)
        self.predicted_proba = predicted_proba

    def accuracy(self):
        return accuracy_score(self.y_true, self.y_pred)

    def expected_calibration_error(self, bins=20):
        return netcal.metrics.ECE(bins).measure(self.predicted_proba, self.y_true)

    def average_calibration_error(self, bins=20):
        return netcal.metrics.ACE(bins).measure(self.predicted_proba, self.y_true)

    def mean_calibration_error(self, bins=20):
        return netcal.metrics.MCE(bins).measure(self.predicted_proba, self.y_true)
