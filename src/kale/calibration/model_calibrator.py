import numpy as np

from kale.models import CalibratableModel


class ModelCalibrator:
    def __init__(self, calibratable_model: CalibratableModel, X_calibrate: np.ndarray, y_calibrate: np.ndarray,
                 X_fit: np.ndarray = None, y_fit: np.ndarray = None):
        self.calibratable_model = calibratable_model
        self.X_calibrate = X_calibrate
        self.y_calibrate = y_calibrate
        self.X_fit = X_fit
        self.y_fit = y_fit

    def set_validation_data(self, X: np.ndarray, y: np.ndarray):
        self.X_fit, self.y_fit = X, y

    def calibrate(self, fit: bool = False):
        if fit:
            if self.X_fit is None or self.y_fit is None:
                raise TypeError("No validation set provided.")
            X, y = self.X_fit, self.y_fit
        else:
            X, y = self.X_calibrate, self.y_calibrate

        self.calibratable_model.fit(X, y)

    def __str__(self):
        return self.__class__.__name__
