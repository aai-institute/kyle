from abc import ABC, abstractmethod
from typing import Generic, List, Optional, TypeVar

import netcal.binning as bn
import netcal.scaling as scl
import numpy as np
from netcal import AbstractCalibration
from sklearn.base import BaseEstimator


class BaseCalibrationMethod(ABC, BaseEstimator):
    @abstractmethod
    def fit(self, confidences: np.ndarray, ground_truth: np.ndarray):
        pass

    @abstractmethod
    def get_calibrated_confidences(self, confidences: np.ndarray):
        pass

    def __str__(self):
        return self.__class__.__name__


def _get_confidences_from_netcal_calibrator(
    confidences: np.ndarray, calibrator: AbstractCalibration
):
    calibrated_confs = calibrator.transform(confidences)

    # TODO: there is a whole bunch of hacks here. I want to get rid of netcal, don't like the code there
    # unfortunately, for 2-dim input netcal gives only the probabilities for the second class,
    # changing the dimension of the output array
    if calibrated_confs.ndim < 2:
        second_class_confs = calibrated_confs
        first_class_confs = 1 - second_class_confs
        calibrated_confs = np.stack([first_class_confs, second_class_confs], axis=1)

        if (
            len(confidences) == 1
        ):  # Netcal has a bug for single data points, this is a dirty fix
            calibrated_confs = calibrated_confs[None, 0]

        if calibrated_confs.shape != confidences.shape:
            raise RuntimeError(
                f"Shape mismatch for input {confidences}, output {calibrated_confs}. "
                f"Netcal output: {second_class_confs}"
            )

    return calibrated_confs


TNetcalModel = TypeVar("TNetcalModel", bound=AbstractCalibration)


# TODO: this is definitely not the final class structure. For now its ok, I want to completely decouple from netcal soon
class NetcalBasedCalibration(BaseCalibrationMethod, Generic[TNetcalModel]):
    def __init__(self, netcal_model: TNetcalModel):
        self.netcal_model = netcal_model

    def fit(self, confidences: np.ndarray, ground_truth: np.ndarray):
        self.netcal_model.fit(confidences, ground_truth)

    def get_calibrated_confidences(self, confidences: np.ndarray) -> np.ndarray:
        return _get_confidences_from_netcal_calibrator(confidences, self.netcal_model)


class TemperatureScaling(NetcalBasedCalibration[scl.TemperatureScaling]):
    def __init__(self):
        super().__init__(scl.TemperatureScaling())


class BetaCalibration(NetcalBasedCalibration[scl.BetaCalibration]):
    def __init__(self):
        super().__init__(scl.BetaCalibration())


class LogisticCalibration(NetcalBasedCalibration[scl.LogisticCalibration]):
    def __init__(self):
        super().__init__(scl.LogisticCalibration())


class IsotonicRegression(NetcalBasedCalibration[bn.IsotonicRegression]):
    def __init__(self):
        super().__init__(bn.IsotonicRegression())


class HistogramBinning(NetcalBasedCalibration[bn.HistogramBinning]):
    def __init__(self, bins=20):
        super().__init__(bn.HistogramBinning(bins=20))


class ClassWiseCalibration(BaseCalibrationMethod):
    def __init__(self, calibration_method_factory=TemperatureScaling):
        self.calibration_method_factory = calibration_method_factory
        self.n_classes: Optional[int] = None
        self.calibration_methods: Optional[List[BaseCalibrationMethod]] = None

    # TODO: maybe parallelize this and predict
    def fit(self, confidences: np.ndarray, labels: np.ndarray):
        self.n_classes = confidences.shape[1]
        self.calibration_methods = []
        for class_label in range(self.n_classes):
            calibration_method = self.calibration_method_factory()
            selected_confs, selected_labels = get_class_confs_labels(
                class_label, confidences, labels
            )
            calibration_method.fit(selected_confs, selected_labels)
            self.calibration_methods.append(calibration_method)

    def get_calibrated_confidences(self, confs: np.ndarray):
        result = np.zeros(confs.shape)
        argmax = confs.argmax(1)
        for class_label in range(self.n_classes):
            scaler = self.calibration_methods[class_label]
            indices = argmax == class_label
            selected_confs = confs[indices]
            calibrated_confs = scaler.get_calibrated_confidences(selected_confs)
            assert calibrated_confs.shape == selected_confs.shape, (
                f"Expected shape {selected_confs.shape} but got {calibrated_confs.shape}. Confs: "
                f"{selected_confs}, output: {calibrated_confs}"
            )

            result[indices] = calibrated_confs
        return result


class ConfidenceReducedCalibration(BaseCalibrationMethod, BaseEstimator):
    def __init__(self, calibration_method=TemperatureScaling()):
        self.calibration_method = calibration_method

    def fit(self, confidences: np.ndarray, ground_truth: np.ndarray):
        reduced_confs, reduced_gt = get_binary_classification_data(
            confidences, ground_truth
        )
        self.calibration_method.fit(reduced_confs, reduced_gt)

    def get_calibrated_confidences(self, confidences: np.ndarray):
        reduced_confs = get_reduced_confidences(confidences)
        reduced_predictions = self.calibration_method.get_calibrated_confidences(
            reduced_confs
        )
        reduced_predictions = reduced_predictions[:, 0]  # take only 0-class prediction
        n_classes = confidences.shape[1]
        non_predicted_class_confidences = (1 - reduced_predictions) / (n_classes - 1)

        # using broadcasting here
        calibrated_confidences = (
            non_predicted_class_confidences * np.ones(confidences.shape).T
        )
        calibrated_confidences = calibrated_confidences.T

        argmax_indices = np.expand_dims(confidences.argmax(axis=1), axis=1)
        np.put_along_axis(
            calibrated_confidences, argmax_indices, reduced_predictions[:, None], axis=1
        )
        assert np.all(
            np.isclose(calibrated_confidences.sum(1), np.ones(len(confidences)))
        )
        assert calibrated_confidences.shape == confidences.shape
        return calibrated_confidences


def get_class_confs_labels(c: int, confidences: np.ndarray, labels: np.ndarray):
    indices = confidences.argmax(1) == c
    return confidences[indices], labels[indices]


def get_reduced_confidences(confidences: np.ndarray):
    top_class_predictions = confidences.max(axis=1)
    return np.stack([top_class_predictions, 1 - top_class_predictions], axis=1)


def get_binary_classification_data(confidences: np.ndarray, labels: np.ndarray):
    new_confidences = get_reduced_confidences(confidences)
    pred_was_correct = labels == confidences.argmax(axis=1)
    # this is a hack - we predict class 0 if pred was correct, else class 1
    new_gt = (np.logical_not(pred_was_correct)).astype(int)
    return new_confidences, new_gt
