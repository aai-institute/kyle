from typing import Union
import numpy as np

from netcal.metrics import MCE as netcal_MCE

from kale.metrics.base_calibration_error import BaseCalibrationError


class MCE(BaseCalibrationError):
    """Maximum Calibration Error. Wraps around netcal's implementation - for further reading refer to netcal's docs."""
    def __init__(self, bins: int = 10):
        super(MCE, self).__init__()
        self.netcal_mce = netcal_MCE(bins=bins)

    def _compute(self, confidences: np.ndarray, ground_truth: np.ndarray, **kwargs)\
            -> Union[float, np.ndarray, ValueError]:
        return self.netcal_mce.measure(confidences, ground_truth, **kwargs)
