import netcal.metrics
import numpy as np
from sklearn.metrics import accuracy_score


class EvalStats:
    """
    Class for computing evaluation statistics of classifiers, including calibration metrics

    :param y_true: integer array of shape (n_samples,)
    :param confidence_vectors: array of shape (n_samples, n_classes)
    :param bins: on how many homogeneous bins to evaluate the statistics
    """
    def __init__(self, y_true: np.ndarray, confidence_vectors: np.ndarray, bins=20):
        assert len(y_true.shape) == 1, f"y_true has to be 1-dimensional, instead got shape: {y_true.shape}"
        assert len(confidence_vectors.shape) == 2 and confidence_vectors.shape[0] == len(y_true), \
            f"predicted_probabilities have to be of shape (#samples, #classes), instead got {confidence_vectors.shape}"
        self.n_samples = len(y_true)
        self.y_true = y_true
        self.y_pred = confidence_vectors.argmax(axis=1)
        self.predicted_proba = confidence_vectors
        
        self.bins = None
        self._bin_boundaries = None
        self.set_bins(bins)
    
    def set_bins(self, bins: int):
        self.bins = bins
        self._bin_boundaries = np.linspace(0, 1, self.bins + 1)
        self._bin_boundaries[0] = -1  # in order to associate predicted probabilities = 0 to the right bin

    def accuracy(self):
        return accuracy_score(self.y_true, self.y_pred)

    def expected_calibration_error(self):
        return netcal.metrics.ECE(self.bins).measure(self.predicted_proba, self.y_true)

    def average_calibration_error(self):
        return netcal.metrics.ACE(self.bins).measure(self.predicted_proba, self.y_true)

    def max_calibration_error(self):
        return netcal.metrics.MCE(self.bins).measure(self.predicted_proba, self.y_true)
    
    def marginal_reliability_hist(self, class_label):
        class_confidences = self.predicted_proba[:, class_label]
        # subtract 1 because with right=True bin 0 corresponds to values < -1 (the first boundary value)
        binned_confidences = np.digitize(x=class_confidences, bins=self._bin_boundaries, right=True) - 1

        marginal_probabilities_per_confidence_bin = []
        for confidence_bin in range(self.bins):
            cur_gt_labels = self.y_true[binned_confidences == confidence_bin]  # only consider gt in current bin
            if len(cur_gt_labels) == 0:
                cur_marginal_probability = 0
            else:
                cur_marginal_probability = np.count_nonzero(cur_gt_labels == class_label)/len(cur_gt_labels)
            marginal_probabilities_per_confidence_bin.append(cur_marginal_probability)
        return np.arange(self.bins) / self.bins, np.array(marginal_probabilities_per_confidence_bin)

    def top_class_reliability_hist(self):
        top_class_confidences = np.choose(self.y_pred, self.predicted_proba.T)
        binned_confidences = np.digitize(x=top_class_confidences, bins=self._bin_boundaries, right=True) - 1

        reliabilities = []
        for confidence_bin in range(self.bins):
            cur_gt_labels = self.y_true[binned_confidences == confidence_bin]  # only consider gt in current bin
            if len(cur_gt_labels) == 0:
                cur_probability = 0
            else:
                cur_pred_labels = self.y_pred[binned_confidences == confidence_bin]
                cur_probability = accuracy_score(cur_gt_labels, cur_pred_labels)
            reliabilities.append(cur_probability)
        return np.arange(self.bins) / self.bins, np.array(reliabilities)
