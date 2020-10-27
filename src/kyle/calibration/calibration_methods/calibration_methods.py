from abc import ABC, abstractmethod

import numpy as np

from netcal.scaling import TemperatureScaling as netcal_TemperatureScaling


class BaseCalibrationMethod(ABC):
    @abstractmethod
    def fit(self, confidences, ground_truth):
        pass

    @abstractmethod
    def get_calibrated_confidences(self, confidences):
        pass

    def __str__(self):
        return self.__class__.__name__


class TemperatureScaling(BaseCalibrationMethod):
    def __init__(self):
        self.netcal_temp_scaling = netcal_TemperatureScaling()

    def fit(self, confidences: np.ndarray, ground_truth: np.ndarray):
        self.netcal_temp_scaling.fit(confidences, ground_truth)

    def get_calibrated_confidences(self, confidences: np.ndarray) -> np.ndarray:
        calibrated_confs = self.netcal_temp_scaling.transform(confidences)

        if calibrated_confs.ndim < 2:
            calibrated_confs = np.vstack((np.subtract(1, calibrated_confs), calibrated_confs)).T

        return calibrated_confs