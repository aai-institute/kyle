from sklearn.base import BaseEstimator

import numpy as np

from kyle.calibration.calibration_methods import (
    BaseCalibrationMethod,
    TemperatureScaling,
)


class CalibratableModel(BaseEstimator):
    def __init__(
        self,
        model: BaseEstimator,
        calibration_method: BaseCalibrationMethod = TemperatureScaling(),
    ):
        self.model = model
        self.calibration_method = calibration_method

    def fit(self, X: np.ndarray, y: np.ndarray):
        uncalibrated_confidences = self.model.predict_proba(X)
        self.calibration_method.fit(uncalibrated_confidences, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        calibrated_proba = self.predict_proba(X)

        return np.argmax(calibrated_proba, axis=2)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        uncalibrated_confidences = self.model.predict_proba(X)
        calibrated_confidences = self.calibration_method.get_calibrated_confidences(
            uncalibrated_confidences
        )

        return calibrated_confidences

    def __str__(self):
        return f"{self.__class__.__name__}, method: {self.calibration_method}"
