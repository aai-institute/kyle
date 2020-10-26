from abc import abstractmethod

import numpy as np
from netcal.metrics import ACE as netcal_ACE, ECE as netcal_ECE, MCE as netcal_MCE


class BaseCalibrationError:

    def __init__(self):
        pass

    @abstractmethod
    def _compute(self, confidences: np.ndarray, ground_truth: np.ndarray, **kwargs) \
            -> float:
        pass

    def compute(self, confidences: np.ndarray, ground_truth: np.ndarray, **kwargs):
        if not np.allclose(np.sum(confidences, axis=1), 1.0, rtol=0.01):
            raise ValueError("Confidences invalid. Probabilities should sum to one.")
        return self._compute(confidences, ground_truth, **kwargs)


class ACE(BaseCalibrationError):
    """Average Calibration Error. Wraps around netcal's implementation - for further reading refer to netcal's docs."""

    def __init__(self, bins: int = 10):
        super(ACE, self).__init__()
        self.netcal_ace = netcal_ACE(bins=bins)

    def _compute(self, confidences: np.ndarray, ground_truth: np.ndarray, **kwargs) \
            -> float:
        return self.netcal_ace.measure(confidences, ground_truth, **kwargs)


class ECE(BaseCalibrationError):
    """Expected Calibration Error. Wraps around netcal's implementation - for further reading refer to netcal's docs."""

    def __init__(self, bins: int = 10):
        super(ECE, self).__init__()
        self.netcal_ece = netcal_ECE(bins=bins)

    def _compute(self, confidences: np.ndarray, ground_truth: np.ndarray, **kwargs) \
            -> float:
        return self.netcal_ece.measure(confidences, ground_truth, **kwargs)


class MCE(BaseCalibrationError):
    """Maximum Calibration Error. Wraps around netcal's implementation - for further reading refer to netcal's docs."""

    def __init__(self, bins: int = 10):
        super(MCE, self).__init__()
        self.netcal_mce = netcal_MCE(bins=bins)

    def _compute(self, confidences: np.ndarray, ground_truth: np.ndarray, **kwargs) \
            -> float:
        return self.netcal_mce.measure(confidences, ground_truth, **kwargs)
