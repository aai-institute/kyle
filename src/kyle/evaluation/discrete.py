from typing import Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from kyle.util import safe_accuracy_score


class EvalStats:
    TOP_CLASS_LABEL = "top_class"

    """
    Class for computing evaluation statistics of classifiers, including calibration metrics

    :param y_true: integer array of shape (n_samples,)
    :param confidences: array of shape (n_samples, n_classes)
    :param bins: on how many homogeneous bins to evaluate the statistics
    """

    def __init__(self, y_true: np.ndarray, confidences: np.ndarray, bins=30):
        assert (
            len(y_true.shape) == 1
        ), f"y_true has to be 1-dimensional, instead got shape: {y_true.shape}"
        assert len(confidences.shape) == 2 and confidences.shape[0] == len(
            y_true
        ), f"predicted_probabilities have to be of shape (#samples, #classes), instead got {confidences.shape}"
        self.num_samples = len(y_true)
        self.num_classes = confidences.shape[1]
        self.y_true = y_true
        self.y_pred = confidences.argmax(axis=1)
        self.confidences = confidences

        self.bins: int = None
        # due to discretization they don't sum to 1 anymore
        self.discretized_confidences: np.ndarray = None
        self.discretized_probab_values: np.ndarray = None
        self.set_bins(bins)

    def discretized_top_class_confidences(self) -> np.ndarray:
        """
        This is essentially the same as self.discretized_confidences.max(axis=1), up to ambiguities in maximum
        due to collisions and discretization.
        We avoid computing max and resolve collisions since we already have computed the argmax before.

        :return: array of shape (N_samples, )
        """
        return np.choose(self.y_pred, self.discretized_confidences.T)

    def set_bins(self, bins: int):
        self.bins = bins
        self.discretized_probab_values = (np.arange(self.bins) + 0.5) / self.bins
        bin_boundaries = np.linspace(0, 1, self.bins + 1)
        bin_boundaries[
            0
        ] = -1  # in order to associate predicted probabilities = 0 to the right bin
        binned_confidences = (
            np.digitize(x=self.confidences, bins=bin_boundaries, right=True) - 1
        )
        binned_confidences = binned_confidences
        self.discretized_confidences = (binned_confidences + 0.5) / self.bins

    def accuracy(self):
        return safe_accuracy_score(self.y_true, self.y_pred)

    def expected_top_class_confidence(self):
        return self.discretized_top_class_confidences().mean()

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

    def marginal_expected_top_class_confidence(self, class_label: int):
        """
        Corresponds to a term involved in ECE_i, given by the integral in formula (TODO)

        :param class_label:
        :return:
        """
        class_label_mask = self.y_pred == class_label
        marginal_top_class_confidences = self.discretized_top_class_confidences()[
            class_label_mask
        ]
        return marginal_top_class_confidences.sum() / len(self.y_true)

    def _expected_error(
        self, probabilities: np.ndarray, members_per_bin: np.ndarray
    ) -> float:
        """
        Computes the expected error, being the sum of abs. differences of true probabilities and mean confidences
        for each bin weighted by the factor N_bin / N_total

        :param probabilities:
        :param members_per_bin:
        :return:
        """
        total_members = np.sum(members_per_bin)
        if total_members == 0:
            return 0.0
        return (
            np.sum(
                np.abs(probabilities - self.discretized_probab_values) * members_per_bin
            )
            / total_members
        )

    def _non_degenerate_acc_conf_differences(self) -> np.ndarray:
        """
        Computes the absolute differences between accuracy and mean confidence for each non-degenerate bin
        where a bin is considered degenerate if for no confidence vector the maximum lies in the bin.
        E.g. for a N-classes classifier, all bins with right-hand value below 1/N will be degenerate since the
        maximum of a probabilities vector is always larger than 1/N.

        :return: array of shape (N_bins, )
        """
        accuracies, members_per_bin = self.top_class_reliabilities()
        acc_conf_difference = (accuracies - self.discretized_probab_values)[
            members_per_bin > 0
        ]
        return np.abs(acc_conf_difference)

    def expected_calibration_error(self):
        accuracies, members_per_bin = self.top_class_reliabilities()
        return self._expected_error(accuracies, members_per_bin)

    def average_calibration_error(self):
        return np.mean(self._non_degenerate_acc_conf_differences())

    def max_calibration_error(self):
        return np.max(self._non_degenerate_acc_conf_differences())

    def expected_marginal_calibration_error(self, class_label):
        """
        I sort of made this up, although this very probably exists somewhere in the wild
        :param class_label:
        """
        class_probabilities, members_per_bin = self.marginal_reliabilities(class_label)
        return self._expected_error(class_probabilities, members_per_bin)

    def average_marginal_calibration_error(self):
        """
        I made this up, don't know if this metric was described anywhere yet.
        It is also not completely clear what this means in terms of probabilistic quantities.
        """
        errors = np.zeros(self.num_classes)
        weights = np.zeros(self.num_classes)
        for class_label in range(self.num_classes):
            accuracies, n_members = self.marginal_reliabilities(class_label)
            total_members = np.sum(n_members)
            errors[class_label] = self._expected_error(accuracies, n_members)
            weights[class_label] = total_members
        return np.sum(errors * weights) / np.sum(weights)

    def marginal_reliabilities(self, class_label: int):
        """
        Compute the true class probabilities and numbers of members (weights) for each of the N bins for the
        confidence for the given class.

        :return: tuple of two 1-dim arrays of length N, corresponding to (accuracy_per_bin, num_members_per_bin)
        """
        class_confidences = self.discretized_confidences[:, class_label]

        members_per_bin = np.zeros(self.bins)
        accuracies_per_bin = np.zeros(self.bins)
        for i, probability_bin in enumerate(self.discretized_probab_values):
            probability_bin_mask = class_confidences == probability_bin
            cur_gt_labels = self.y_true[probability_bin_mask]

            cur_members = np.sum(probability_bin_mask)
            cur_accuracy = safe_accuracy_score(
                cur_gt_labels, class_label * np.ones(len(cur_gt_labels))
            )
            members_per_bin[i] = cur_members
            accuracies_per_bin[i] = cur_accuracy
        return accuracies_per_bin, members_per_bin

    def top_class_reliabilities(self):
        """
        Compute the accuracies and numbers of members (weights) for each of the N bins for top-class confidence.

        :return: tuple of two 1-dim arrays of length N, corresponding to (accuracy_per_bin, num_members_per_bin)
        """
        members_per_bin = np.zeros(self.bins)
        accuracies_per_bin = np.zeros(self.bins)
        for i, probability in enumerate(self.discretized_probab_values):
            probability_bin_mask = (
                self.discretized_top_class_confidences() == probability
            )
            cur_gt_labels = self.y_true[probability_bin_mask]
            cur_pred_labels = self.y_pred[probability_bin_mask]

            cur_members = np.sum(probability_bin_mask)
            cur_accuracy = safe_accuracy_score(cur_gt_labels, cur_pred_labels)
            members_per_bin[i] = cur_members
            accuracies_per_bin[i] = cur_accuracy
        return accuracies_per_bin, members_per_bin

    def plot_reliability_curves(
        self, class_labels: Sequence[Union[int, str]], display_weights=False
    ):
        """

        :param class_labels:
        :param display_weights: If True, for each reliability curve the weights of each bin will be
            plotted as histogram. The weights have been scaled for the sake of display, only relative differences
            between them have an interpretable meaning.
            The errors containing "expected" in the name take these weights into account.
        :return:
        """
        colors = ListedColormap(["y", "g", "r", "c", "m"])

        plt.figure()
        plt.title(f"Reliability curves ({self.bins} bins)")
        plt.xlabel("confidence")
        plt.ylabel("ground truth probability")
        plt.axis("equal")
        x_values = self.discretized_probab_values
        plt.plot(
            np.linspace(0, 1), np.linspace(0, 1), label="perfect calibration", color="b"
        )
        for i, class_label in enumerate(class_labels):
            color = colors(i)
            if isinstance(class_label, int):
                label = f"class {class_label}"
                y_values, weights = self.marginal_reliabilities(class_label)
            elif class_label == self.TOP_CLASS_LABEL:
                label = "prediction"
                y_values, weights = self.top_class_reliabilities()
            else:
                raise ValueError(f"Unknown class label: {class_label}")
            plt.plot(x_values, y_values, marker=".", label=label, color=color)
            if display_weights:
                # rescale the weights such that the maximum is at 1/2 for improved visibility
                weights = 1 / 2 * weights / weights.max()
                plt.bar(x_values, weights, alpha=0.2, width=1 / self.bins, color=color)

        axes = plt.gca()
        axes.set_xlim([0, 1])
        axes.set_ylim([0, 1])
        plt.legend(loc="best")
        plt.show()

    def plot_confidence_distributions(
        self, class_labels: Sequence[Union[int, str]], new_fig=True
    ):
        """

        :param new_fig:
        :param class_labels:
        :return:
        """
        colors = ListedColormap(["y", "g", "r", "c", "m"])

        if new_fig:
            plt.figure()
        plt.title(f" Confidence Distribution ({self.bins} bins)")
        plt.xlabel("confidence")
        plt.ylabel("Frequency")
        x_values = self.discretized_probab_values

        for i, class_label in enumerate(class_labels):
            color = colors(i)
            if isinstance(class_label, int):
                label = f"class {class_label}"
                _, weights = self.marginal_reliabilities(class_label)
            elif class_label == self.TOP_CLASS_LABEL:
                label = "prediction"
                _, weights = self.top_class_reliabilities()
            else:
                raise ValueError(f"Unknown class label: {class_label}")
            plt.bar(
                x_values,
                weights,
                alpha=0.3,
                width=1 / self.bins,
                label=label,
                color=color,
            )

        axes = plt.gca()
        axes.set_xlim([0, 1])
        plt.legend(loc="best")
        if new_fig:
            plt.show()
