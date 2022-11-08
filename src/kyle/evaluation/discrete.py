import logging
from typing import Dict, Literal, NamedTuple, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.utils import check_consistent_length, column_or_1d

from kyle.util import safe_accuracy_score

log = logging.getLogger(__name__)


class ReliabilityResult(NamedTuple):
    prob_true: np.ndarray
    prob_pred: np.ndarray
    n_members: np.ndarray
    bin_edges: np.ndarray


def _assert_1d(arr: np.ndarray, arr_name: str, error_type=ValueError):
    if arr.ndim != 1:
        raise error_type(f"{arr_name} should be a 1d array but got shape: {arr.shape}")


def binary_classifier_reliability(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins=12,
    strategy: Literal["uniform", "quantile"] = "uniform",
) -> ReliabilityResult:
    """
    The implementation is essentially a copy of `sklearn.calibration.calibration_curve` but contains additional
    quantities in the return: `n_members` and `bin_edges`. Unfortunately, the sklearn implementation cannot be
    used as it doesn't return these quantities, and they would have to be recomputed.
    Compute arrays related to the reliabilities of a binary classifiers. They can be used e.g. for computing
    calibration errors or for visualizing reliability curves.

    :param n_bins:
    :param y_true: the confidences in the prediction of the class **True**
        (or class 1, if labels passed as integers). Should be a numpy array of shape (n_samples, )
    :param y_prob: array of shape (n_samples, ) containing True and False or integers (1, 0) respectively
    :param strategy: Strategy used to define the widths of the bins.
        **uniform** The bins have identical widths.
        **quantile** The bins have the same number of samples and depend on y_prob.
    :return: named tuple containing arrays with confidences, accuracies, members, bin_edges
    """
    y_true = column_or_1d(y_true)
    y_prob = column_or_1d(y_prob)
    check_consistent_length(y_true, y_prob)

    if y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError("y_prob has values outside [0, 1].")

    uniform_bins = np.linspace(0.0, 1.0, n_bins + 1)
    if strategy == "quantile":  # Determine bin edges by distribution of data
        bins = np.quantile(y_prob, uniform_bins)
    elif strategy == "uniform":
        bins = uniform_bins
    else:
        raise ValueError(
            "Invalid entry to 'strategy' input. Strategy "
            "must be either 'quantile' or 'uniform'."
        )

    binids = np.searchsorted(bins[1:-1], y_prob)

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    is_nonempty = bin_total != 0
    n_empty_bins = n_bins - is_nonempty.sum()
    if n_empty_bins > 0:
        log.info(
            f"{n_empty_bins} of {n_bins} bins were empty, the reliability curve cannot be estimated in them."
            f"This can be prevented by either: \n"
            f"  1) reducing the number of bins (current value is {n_bins}) or \n"
            f"  2) increasing the sample size (current value is {len(y_true)}) or \n"
            f"  3) using strategy='quantile'"
        )

    last_nonempty_bin = np.where(is_nonempty)[0][-1]
    if last_nonempty_bin == n_bins:
        last_bin_edge = 1.0
    else:
        last_bin_edge = bins[last_nonempty_bin + 1]

    bin_edges = np.append(bins[is_nonempty], last_bin_edge)
    bin_members = bin_total[is_nonempty]
    prob_true = bin_true[is_nonempty] / bin_members
    prob_pred = bin_sums[is_nonempty] / bin_members

    return ReliabilityResult(prob_true, prob_pred, bin_members, bin_edges)


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
        assert (
            confidences.ndim == 2
        ), f"confidences have to be of shape (#samples, #classes), instead got {confidences.shape}"
        assert confidences.shape[0] == len(
            y_true
        ), f"Mismatch between number of data points in confidences and labels, {confidences.shape[0]} != {len(y_true)}"
        self.y_true = y_true
        self.confidences = confidences

        # Saving some fields, so they don't have to be recomputed
        self.num_samples = len(y_true)
        self.num_classes = confidences.shape[1]
        self.y_pred = confidences.argmax(axis=1)
        # noinspection PyArgumentList
        self._top_class_confidences = confidences.max(axis=1)
        self._argmax_predicted_mask = self.y_true == self.y_pred
        # reshaping b/c the mask can be computed through numpy broadcasting
        possible_labels = np.arange(self.num_classes).reshape((self.num_classes, 1))
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

    def marginal_accuracy(self, class_label: int):
        """
        Corresponds to acc_i in our calibration paper

        :param class_label:
        :return:
        """
        class_label_mask = self.y_pred == class_label
        predictions = self.y_pred[class_label_mask]
        gt = self.y_true[class_label_mask]
        return np.sum(gt == predictions) / len(self.y_true)

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
        if class_label == "top_class":
            reduced_y_pred = self.top_class_confidences
            reduced_y_true = self._argmax_predicted_mask
        else:
            reduced_y_pred = self.confidences[:, class_label]
            reduced_y_true = self._label_predicted_masks[class_label]
        return binary_classifier_reliability(
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
        colors = ListedColormap(["y", "g", "r", "c", "m"])

        fig = plt.figure()
        plt.title(f"Reliability curves ({n_bins} bins)")
        plt.xlabel("confidence")
        plt.ylabel("ground truth probability")
        plt.axis("equal")

        # plotting a diagonal for perfect calibration
        plt.plot([0, 1], [0, 1], label="perfect calibration", color="b")

        # for each class, plot curve and weights, cycle through colors
        for i, class_label in enumerate(class_labels):
            color = colors(i)
            if class_label == "top_class":
                plot_label = "prediction"
            else:
                plot_label = f"class {class_label}"

            prob_true, prob_pred, n_members, bin_edges = self.reliabilities(
                class_label,
                n_bins=n_bins,
                strategy=strategy,
            )
            plt.plot(prob_pred, prob_true, marker=".", label=plot_label, color=color)
            if display_weights:
                # rescale the weights for improved visibility
                weights = n_members / (3 * n_members.max())
                width = np.diff(bin_edges)
                plt.bar(
                    bin_edges[:-1],
                    weights,
                    align="edge",
                    alpha=0.2,
                    width=width,
                    color=color,
                    label=f"weights ({plot_label})",
                    edgecolor="black",
                    linewidth=0.5,
                    linestyle="--",
                )

        axes = plt.gca()
        axes.set_xlim([0, 1])
        axes.set_ylim([0, 1])
        plt.legend(loc="best")
        return fig

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
