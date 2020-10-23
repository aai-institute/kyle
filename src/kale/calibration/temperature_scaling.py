from kale.calibration.base_calibration import BaseCalibration


class TemperatureScaling(BaseCalibration):
    def __init__(self):
        super().__init__()
        self.temperature = TemperatureScaling()

    def fit(self, X, y):
        self.temperature.fit(X, y)

    def adjust_confidences(self, confidences):
        self.temperature.transform(confidences)
