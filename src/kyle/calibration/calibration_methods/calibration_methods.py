from abc import ABC, abstractmethod

import numpy as np

from netcal.scaling import TemperatureScaling as netcal_TemperatureScaling


class BaseCalibrationMethod(ABC):
    @abstractmethod
    def fit(self, confidences: np.ndarray, ground_truth: np.ndarray):
        pass

    @abstractmethod
    def get_calibrated_confidences(self, confidences: np.ndarray):
        pass

    def __str__(self):
        return self.__class__.__name__


class TemperatureScaling(BaseCalibrationMethod):
    """
    Temperature scaling technique to calibrate classifiers [1]_. A variant of Platt scaling [2]_, temperature scaling
    is a post-processing method that learns a scalar parameter to adjust uncalibrated probabilities.

    References
    ----------
    .. [1] Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks.
    .. [2] Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized
            likelihood methods. Advances in large margin classifiers, 10(3), 61-74.
    """

    def __init__(self):
        self.netcal_temp_scaling = netcal_TemperatureScaling()

    def fit(self, confidences: np.ndarray, ground_truth: np.ndarray):
        self.netcal_temp_scaling.fit(confidences, ground_truth)

    def get_calibrated_confidences(self, confidences: np.ndarray) -> np.ndarray:
        calibrated_confs = self.netcal_temp_scaling.transform(confidences)

        # unfortunately, for 2-dim input netcal gives only the probabilities for the second class,
        # changing the dimension of the output array
        if calibrated_confs.ndim < 2:
            second_class_confs = calibrated_confs
            first_class_confs = 1 - second_class_confs
            calibrated_confs = np.vstack((first_class_confs, second_class_confs)).T

        return calibrated_confs