import numpy as np
from sklearn.base import BaseEstimator

from kale.calibration.base_calibration import BaseCalibration


class CalibratedModel(BaseEstimator):
    def __init__(self, model: BaseEstimator, calibration_method: BaseCalibration):
        self.model = model
        self.calibration_method = calibration_method

    def calibrate(self, X, y) -> None:
        uncalibrated_confidences = self.model.predict_proba(X)
        self.calibration_method.fit(uncalibrated_confidences, y)

    def fit(self, X, y) -> None:
        self.model.fit(X, y)

    def predict(self, X) -> np.ndarray:
        uncalibrated_confidences = self.model.predict_proba(X)
        calibrated_confidences = self.calibration_method.adjust_confidences(uncalibrated_confidences)

        if calibrated_confidences.ndim < 2:
            calibrated_confidences = np.vstack((np.subtract(1, calibrated_confidences), calibrated_confidences)).T

        return calibrated_confidences
