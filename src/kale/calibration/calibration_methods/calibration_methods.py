from abc import ABC, abstractmethod

import numpy as np

from netcal.scaling import TemperatureScaling as netcal_TemperatureScaling


class BaseCalibrationMethod(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def get_calibrated_confidences(self, confidences):
        pass

    def __str__(self):
        return self.__class__.__name__


class TemperatureScaling(BaseCalibrationMethod):
    def __init__(self):
        self.netcal_temp_scaling = netcal_TemperatureScaling()

    def fit(self, X, y):
        self.netcal_temp_scaling.fit(X, y)

    def get_calibrated_confidences(self, confidences) -> np.ndarray:
        return self.netcal_temp_scaling.transform(confidences)
