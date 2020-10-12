from typing import Union
import numpy as np

from netcal.metrics import MCE as netcal_MCE

from metrics.base_calibration_error import BaseCalibrationError


class MCE(BaseCalibrationError):
    """Maximum Calibration Error. Wraps around netcal's implementation - for further reading refer to netcal's docs."""
    def __init__(self, bins: int = 10):
        super(MCE, self).__init__()
        self.netcal_mce = netcal_MCE(bins=bins)

    def measure(self, confidences: np.ndarray, ground_truth: np.ndarray, **kwargs)\
            -> Union[float, np.ndarray, ValueError]:
        input_is_invalid, error_message = self.check_input_is_invalid(confidences, ground_truth)
        if input_is_invalid:
            raise ValueError(error_message)
        return self.netcal_mce.measure(confidences, ground_truth, **kwargs)