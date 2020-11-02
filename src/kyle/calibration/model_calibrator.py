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

    def calibrate(self, calibratable_model: CalibratableModel, fit: bool = False):
        if fit:
            if self.X_fit is None or self.y_fit is None:
                raise AttributeError("No dataset for fitting provided")
            calibratable_model.fit(self.X_fit, self.y_fit)

        calibratable_model.calibrate(self.X_calibrate, self.y_calibrate)

    def __str__(self):
        return self.__class__.__name__
