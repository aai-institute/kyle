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
        assert (
            len(confidences.shape) == 2
        ), f"predicted_probabilities have to be of shape (#samples, #classes), instead got {confidences.shape}"
        assert confidences.shape[0] == len(
            y_true
        ), f"Mismatch between number of data points in confidences and labels, {confidences.shape[0]} != {len(y_true)}"
        self.num_samples = len(y_true)
        self.num_classes = confidences.shape[1]
        self.y_true = y_true
        self.y_pred = confidences.argmax(axis=1)
        self.confidences = confidences
        self._top_class_confidences = confidences.max(axis=1)

        self.bins: int = None
        # due to discretization they don't sum to 1 anymore
        self._discretized_confidences: np.ndarray = None
        self._discretized_probab_values: np.ndarray = None
        self.set_bins(bins)

    def expected_confidence(self, class_label: Union[int, str] = TOP_CLASS_LABEL):
        """
        Returns the expected confidence for the selected class or for the predictions (default)

        :param class_label: either the class label as int or "top_class"
        :return:
        """
        if class_label == self.TOP_CLASS_LABEL:
            confs = self._top_class_confidences
        else:
            confs = self.confidences[:, class_label]
        return float(np.mean(confs))

    def set_bins(self, bins: int):
        self.bins = bins
        self._discretized_probab_values = (np.arange(self.bins) + 0.5) / self.bins
        bin_boundaries = np.linspace(0, 1, self.bins + 1)
        bin_boundaries[
            0
        ] = -1  # in order to associate predicted probabilities = 0 to the right bin
        binned_confidences = (
            np.digitize(x=self.confidences, bins=bin_boundaries, right=True) - 1
        )
        self._discretized_confidences = (binned_confidences + 0.5) / self.bins

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

    @staticmethod
    def _expected_error(
        probabilities: np.ndarray, members_per_bin: np.ndarray, confidences: np.ndarray
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
        result = float(np.sum(np.abs(probabilities - confidences) * members_per_bin))
        result /= total_members
        return result

    def _non_degenerate_acc_conf_differences(self) -> np.ndarray:
        """
        Computes the absolute differences between accuracy and mean confidence for each non-degenerate bin
        where a bin is considered degenerate if for no confidence vector the maximum lies in the bin.
        E.g. for a N-classes classifier, all bins with right-hand value below 1/N will be degenerate since the
        maximum of a probabilities vector is always larger than 1/N.

        :return: array of shape (N_bins, )
        """
        accuracies, members_per_bin, confidences = self.top_class_reliabilities()
        acc_conf_difference = (accuracies - confidences)[members_per_bin > 0]
        return np.abs(acc_conf_difference)

    def expected_calibration_error(self):
        accuracies, members_per_bin, confidences = self.top_class_reliabilities()
        return self._expected_error(accuracies, members_per_bin, confidences)

    def average_calibration_error(self):
        return np.mean(self._non_degenerate_acc_conf_differences())

    def max_calibration_error(self):
        return np.max(self._non_degenerate_acc_conf_differences())

    def expected_marginal_calibration_error(self, class_label):
        """
        I sort of made this up, although this very probably exists somewhere in the wild
        :param class_label:
        """
        (
            class_probabilities,
            members_per_bin,
            class_confidences,
        ) = self.marginal_reliabilities(class_label)
        return self._expected_error(
            class_probabilities, members_per_bin, class_confidences
        )

    def average_marginal_calibration_error(self):
        """
        I made this up, don't know if this metric was described anywhere yet.
        It is also not completely clear what this means in terms of probabilistic quantities.
        """
        errors = np.zeros(self.num_classes)
        weights = np.zeros(self.num_classes)
        for class_label in range(self.num_classes):
            accuracies, n_members, class_confidences = self.marginal_reliabilities(
                class_label
            )
            total_members = np.sum(n_members)
            errors[class_label] = self._expected_error(
                accuracies, n_members, class_confidences
            )
            weights[class_label] = total_members
        return np.sum(errors * weights) / np.sum(weights)

    def class_wise_expected_calibration_error(self):
        result = sum(
            self.expected_marginal_calibration_error(k) for k in range(self.num_classes)
        )
        result /= self.num_classes
        return result

    def marginal_reliabilities(self, class_label: int):
        """
        Compute the true class probabilities and numbers of members (weights) for each of the N bins for the
        confidence for the given class.

        :return: tuple of two 1-dim arrays of length N, corresponding to (accuracy_per_bin, num_members_per_bin)
        """
        discretized_class_confidences = self._discretized_confidences[:, class_label]
        class_confidences = self.confidences[:, class_label]

        members_per_bin = np.zeros(self.bins)
        accuracies_per_bin = np.zeros(self.bins)
        mean_class_confidences_per_bin = np.zeros(self.bins)
        for i, probability_bin in enumerate(self._discretized_probab_values):
            probability_bin_mask = discretized_class_confidences == probability_bin
            cur_gt_labels = self.y_true[probability_bin_mask]
            cur_class_confidences = class_confidences[probability_bin_mask]

            cur_members = np.sum(probability_bin_mask)
            cur_accuracy = safe_accuracy_score(
                cur_gt_labels, class_label * np.ones(len(cur_gt_labels))
            )
            if len(cur_class_confidences) > 0:
                cur_mean_class_confidence = cur_class_confidences.mean()
            else:
                cur_mean_class_confidence = probability_bin
            members_per_bin[i] = cur_members
            accuracies_per_bin[i] = cur_accuracy
            mean_class_confidences_per_bin[i] = cur_mean_class_confidence
        return accuracies_per_bin, members_per_bin, mean_class_confidences_per_bin

    def top_class_reliabilities(self):
        """
        Compute the accuracies and numbers of members (weights) for each of the N bins for top-class confidence.

        :return: tuple of two 1-dim arrays of length N, corresponding to (accuracy_per_bin, num_members_per_bin)
        """
        members_per_bin = np.zeros(self.bins)
        accuracies_per_bin = np.zeros(self.bins)
        mean_confidences_per_bin = np.zeros(self.bins)
        discretized_top_class_confidences = self._discretized_confidences.max(axis=1)
        for i, probability in enumerate(self._discretized_probab_values):
            probability_bin_mask = discretized_top_class_confidences == probability
            cur_members = np.sum(probability_bin_mask)
            if cur_members == 0:
                members_per_bin[i] = 0
                accuracies_per_bin[i] = 0
                mean_confidences_per_bin[i] = 0
                continue

            cur_gt_labels = self.y_true[probability_bin_mask]
            cur_pred_labels = self.y_pred[probability_bin_mask]
            cur_top_class_confidences = self._top_class_confidences[
                probability_bin_mask
            ]

            cur_accuracy = safe_accuracy_score(cur_gt_labels, cur_pred_labels)
            cur_mean_confidence = cur_top_class_confidences.mean()
            members_per_bin[i] = cur_members
            accuracies_per_bin[i] = cur_accuracy
            mean_confidences_per_bin[i] = cur_mean_confidence
        return accuracies_per_bin, members_per_bin, mean_confidences_per_bin

    # TODO: the reliabilities are plotted above the centers of bins, not above the mean confidences
    #   The latter would plotting multiple curves at once impossible but the plot would be more precise
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
        x_values = self._discretized_probab_values
        plt.plot(
            np.linspace(0, 1), np.linspace(0, 1), label="perfect calibration", color="b"
        )
        for i, class_label in enumerate(class_labels):
            color = colors(i)
            if isinstance(class_label, int):
                label = f"class {class_label}"
                y_values, weights, _ = self.marginal_reliabilities(class_label)
            elif class_label == self.TOP_CLASS_LABEL:
                label = "prediction"
                y_values, weights, _ = self.top_class_reliabilities()
            else:
                raise ValueError(f"Unknown class label: {class_label}")
            plt.plot(x_values, y_values, marker=".", label=label, color=color)
            if display_weights:
                # rescale the weights such that the maximum is at 1/2 for improved visibility
                weights = 1 / 2 * weights / weights.max()
                plt.bar(
                    x_values,
                    weights,
                    alpha=0.2,
                    width=1 / self.bins,
                    color=color,
                    label=f"bin_weights for {label}",
                )

        axes = plt.gca()
        axes.set_xlim([0, 1])
        axes.set_ylim([0, 1])
        plt.legend(loc="best")

    # TODO: delete, I don't think we need this. Maybe add flag to only plot bin weights to the plot above
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
        x_values = self._discretized_probab_values

        for i, class_label in enumerate(class_labels):
            color = colors(i)
            if isinstance(class_label, int):
                label = f"class {class_label}"
                _, weights, _ = self.marginal_reliabilities(class_label)
            elif class_label == self.TOP_CLASS_LABEL:
                label = "prediction"
                _, weights, _ = self.top_class_reliabilities()
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
