import numpy as np
from netcal.metrics import ECE as netcal_ECE

from metrics.base_calibration_error import BaseCalibrationError


class ECE(BaseCalibrationError):
    def __init__(self, bins: int = 10):
        super(ECE, self).__init__()
        self.netcal_ece = netcal_ECE(bins=bins)

    def measure(self, confidences: np.ndarray, ground_truth: np.ndarray, **kwargs):
        error_message = self.validate_input(confidences, ground_truth)
        if error_message is not None:
            raise ValueError(error_message)
        return self.netcal_ece.measure(confidences, ground_truth, **kwargs)