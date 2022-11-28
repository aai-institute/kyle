import logging
from typing import Dict, Literal, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np

from kyle.evaluation.reliabilities import (
    _assert_1d,
    _binary_classifier_reliability,
    _plot_reliability_curves,
    _to2d,
    classifier_reliability,
)
from kyle.util import safe_accuracy_score

log = logging.getLogger(__name__)


class EvalStats:
    """
    Class for computing evaluation statistics of classifiers, including calibration metrics
    """

    def __init__(
        self,
        y_true: np.ndarray,
        confidences: np.ndarray,
    ):
        """

        :param y_true: integer array of shape (n_samples,). Assumed to contain true labels in range [0, n_classes-1]
        :param confidences: array of shape (n_samples, n_classes)
        """
        _assert_1d(y_true, "y_true")
        self.y_true = y_true
        self.confidences = _to2d(confidences)

        # Saving some fields, so they don't have to be recomputed
        self.num_samples = len(y_true)
        self.num_classes = confidences.shape[1]
        self.y_pred = confidences.argmax(axis=1)
        # noinspection PyArgumentList
        self._top_class_confidences = np.take_along_axis(
            confidences, self.y_pred[:, None], axis=1
        ).T[0]
        self._argmax_predicted_mask = self.y_true == self.y_pred
        # reshaping b/c the mask can be computed through numpy broadcasting
        possible_labels = np.arange(self.num_classes).reshape((self.num_classes, 1))
        # from this field we can get a mask for whether the label i was predicted
        # by calling self._label_predicted_masks[i]
        # noinspection PyTypeChecker
        self._label_predicted_masks: np.ndarray = self.y_true == possible_labels

    @property
    def top_class_confidences(self):
        return self._top_class_confidences

    def expected_confidence(
        self, class_label: Union[int, Literal["top_class"]] = "top_class"
    ):
        """
        Returns the expected confidence for the selected class or for the predictions (default)

        :param class_label: either the class label as int or "top_class"
        :return:
        """
        if class_label == "top_class":
            confs = self._top_class_confidences
        else:
            confs = self.confidences[:, class_label]
        return float(np.mean(confs))

    def accuracy(self):
        return safe_accuracy_score(self.y_true, self.y_pred)

    def expected_calibration_error(
        self,
        class_label: Union[int, Literal["top_class"]] = "top_class",
        n_bins=12,
        strategy: Literal["uniform", "quantile"] = "uniform",
    ):
        """
        :param class_label: if "top_class", will be the usual (confidence) ECE. Otherwise, it will be the
            marginal class-wise ECE for the selected class.
        :param n_bins:
        :param strategy:
        :return:
        """
        reliabilities = self.reliabilities(
            class_label=class_label,
            n_bins=n_bins,
            strategy=strategy,
        )
        sum_members = np.sum(reliabilities.n_members)
        if sum_members == 0:
            return 0.0
        weights = reliabilities.n_members / sum_members
        abs_diff = np.abs(reliabilities.prob_pred - reliabilities.prob_true)
        return np.dot(abs_diff, weights)

    def average_calibration_error(
        self, n_bins=12, strategy: Literal["uniform", "quantile"] = "uniform"
    ):
        reliabilities = self.reliabilities(
            class_label="top_class", n_bins=n_bins, strategy=strategy
        )
        abs_distances = np.abs(reliabilities.prob_pred - reliabilities.prob_true)
        return np.mean(abs_distances)

    def max_calibration_error(
        self, n_bins=12, strategy: Literal["uniform", "quantile"] = "uniform"
    ):
        reliabilities = self.reliabilities(
            class_label="top_class", n_bins=n_bins, strategy=strategy
        )
        abs_distances = np.abs(reliabilities.prob_pred - reliabilities.prob_true)
        return np.max(abs_distances)

    def class_wise_expected_calibration_error(
        self, n_bins=12, strategy: Literal["uniform", "quantile"] = "uniform"
    ):
        sum_marginal_errors = sum(
            self.expected_calibration_error(k, n_bins=n_bins, strategy=strategy)
            for k in range(self.num_classes)
        )
        return sum_marginal_errors / self.num_classes

    # TODO or not TODO: could in principle work for any 1-dim. reduction but we might not need this generality
    def reliabilities(
        self,
        class_label: Union[int, Literal["top_class"]],
        n_bins=12,
        strategy: Literal["uniform", "quantile"] = "uniform",
    ):
        """
        Computes arrays related to the reliabilities of the provided confidences. They can be used e.g. for computing
        calibration errors or for visualizing reliability curves.

        :param n_bins:
        :param class_label: either an integer label for the class-wise reliabilities, or "top_class" for the
            reliabilities in predictions.
        :param strategy:

        :return: named tuple containing arrays with confidences, accuracies, members, bin_edges
        """
        # Reducing here to save time on re-computation
        if class_label == "top_class":
            reduced_y_pred = self.top_class_confidences
            reduced_y_true = self._argmax_predicted_mask
        else:
            reduced_y_pred = self.confidences[:, class_label]
            reduced_y_true = self._label_predicted_masks[class_label]
        return _binary_classifier_reliability(
            reduced_y_true,
            reduced_y_pred,
            n_bins=n_bins,
            strategy=strategy,
        )

    def plot_reliability_curves(
        self,
        class_labels: Sequence[Union[int, Literal["top_class"]]],
        display_weights=False,
        n_bins=12,
        strategy: Literal["uniform", "quantile"] = "uniform",
    ):
        """

        :param class_labels:
        :param display_weights: If True, for each reliability curve the weights of each bin will be
            plotted as histogram. The weights have been scaled for the sake of display, only relative differences
            between them have an interpretable meaning.
            The errors containing "expected" in the name take these weights into account.
        :param strategy:
        :param n_bins:
        :return: figure
        """
        return _plot_reliability_curves(
            self.reliabilities, class_labels, display_weights, n_bins, strategy
        )

    def plot_gt_distribution(self, label_names: Dict[int, str] = None):
        class_labels, counts = np.unique(self.y_true, return_counts=True)
        if label_names is not None:
            class_labels = [
                label_names.get(label_id, label_id) for label_id in class_labels
            ]

        fig, ax = plt.subplots()
        ax.pie(counts, labels=class_labels, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax.set_title("Ground Truth Distribution")
        return fig
