import numpy as np

from kyle.models import CalibratableModel


class ModelCalibrator:
    def __init__(
        self,
        X_calibrate: np.ndarray,
        y_calibrate: np.ndarray,
        X_fit: np.ndarray = None,
        y_fit: np.ndarray = None,
    ):
        self.X_calibrate = X_calibrate
        self.y_calibrate = y_calibrate
        self.X_fit = X_fit
        self.y_fit = y_fit

    def set_validation_data(self, X: np.ndarray, y: np.ndarray):
        self.X_fit, self.y_fit = X, y

    def calibrate(self, calibratable_model: CalibratableModel, fit: bool = False):
        if fit:
            if self.X_fit is None or self.y_fit is None:
                raise AttributeError("No validation set provided.")
            calibratable_model.fit(self.X_calibrate, self.y_calibrate)
            X_val, y_val = self.X_fit, self.y_fit
        else:
            X_val, y_val = self.X_calibrate, self.y_calibrate

        calibratable_model.calibrate(X_val, y_val)

        return calibratable_model

    def __str__(self):
        return self.__class__.__name__
