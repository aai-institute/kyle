from typing import Union, Sequence

import matplotlib.pyplot as plt
import netcal.metrics
import numpy as np

from kyle.util import safe_accuracy_score


class EvalStats:
    TOP_CLASS_LABEL = "top_class"

    """
    Class for computing evaluation statistics of classifiers, including calibration metrics

    :param y_true: integer array of shape (n_samples,)
    :param confidences: array of shape (n_samples, n_classes)
    :param bins: on how many homogeneous bins to evaluate the statistics
    """
    def __init__(self, y_true: np.ndarray, confidences: np.ndarray, bins=20):
        assert len(y_true.shape) == 1, f"y_true has to be 1-dimensional, instead got shape: {y_true.shape}"
        assert len(confidences.shape) == 2 and confidences.shape[0] == len(y_true), \
            f"predicted_probabilities have to be of shape (#samples, #classes), instead got {confidences.shape}"
        self.n_samples = len(y_true)
        self.y_true = y_true
        self.y_pred = confidences.argmax(axis=1)
        self.confidences = confidences
        
        self.bins = None
        self.confidence_bins = None
        self.set_bins(bins)
    
    def set_bins(self, bins: int):
        self.bins = bins
        bin_boundaries = np.linspace(0, 1, self.bins + 1)
        bin_boundaries[0] = -1  # in order to associate predicted probabilities = 0 to the right bin
        self.confidence_bins = np.digitize(x=self.confidences, bins=bin_boundaries, right=True) - 1

    def accuracy(self):
        return safe_accuracy_score(self.y_true, self.y_pred)

    def expected_calibration_error(self):
        return netcal.metrics.ECE(self.bins).measure(self.confidences, self.y_true)

    def average_calibration_error(self):
        return netcal.metrics.ACE(self.bins).measure(self.confidences, self.y_true)

    def max_calibration_error(self):
        return netcal.metrics.MCE(self.bins).measure(self.confidences, self.y_true)
    
    def marginal_reliability_hist(self, class_label: int):
        class_confidence_bins = self.confidence_bins[:, class_label]

        accuracies_per_bin = []
        for confidence_bin in range(self.bins):
            cur_gt_labels = self.y_true[class_confidence_bins == confidence_bin]  # only consider gt in current bin
            # cur_accuracy = np.count_nonzero(cur_gt_labels == class_label)/len(cur_gt_labels)
            accuracies_per_bin.append(safe_accuracy_score(cur_gt_labels, class_label * np.ones(len(cur_gt_labels))))
        return np.arange(self.bins) / self.bins, np.array(accuracies_per_bin)

    def top_class_reliability_hist(self):
        top_class_confidence_bins = np.choose(self.y_pred, self.confidence_bins.T)

        accuracies_per_bin = []
        for confidence_bin in range(self.bins):
            cur_gt_labels = self.y_true[top_class_confidence_bins == confidence_bin]  # only consider gt in current bin
            cur_pred_labels = self.y_pred[top_class_confidence_bins == confidence_bin]
            accuracies_per_bin.append(safe_accuracy_score(cur_gt_labels, cur_pred_labels))
        return np.arange(self.bins) / self.bins, np.array(accuracies_per_bin)

    def plot_reliability_curves(self, class_labels: Sequence[Union[int, str]]):
        plt.figure()
        plt.title(f"Reliability curves")
        plt.xlabel("confidence")
        plt.ylabel("ground truth probability")
        plt.axis("equal")
        plt.plot(np.linspace(0, 1), np.linspace(0, 1), label="perfect calibration")
        for class_label in class_labels:
            if isinstance(class_label, int):
                label = f"class {class_label}"
                hist = self.marginal_reliability_hist(class_label)
            elif class_label == self.TOP_CLASS_LABEL:
                label = "prediction"
                hist = self.top_class_reliability_hist()
            else:
                raise ValueError(f"Unknown class label: {class_label}")
            plt.plot(*hist, marker="o", label=label)
        axes = plt.gca()
        axes.set_xlim([0, 1])
        axes.set_ylim([0, 1])
        plt.legend(loc="best")
        plt.show()

