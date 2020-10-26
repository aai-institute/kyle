from abc import abstractmethod, ABC

import netcal.metrics
import numpy as np


class BaseCalibrationError(ABC):

    @abstractmethod
    def _compute(self, confidences: np.ndarray, ground_truth: np.ndarray, **kwargs) \
            -> float:
        pass

    def compute(self, confidences: np.ndarray, ground_truth: np.ndarray, **kwargs):
        if not np.allclose(np.sum(confidences, axis=1), 1.0, rtol=0.01):
            raise ValueError("Confidences invalid. Probabilities should sum to one.")
        return self._compute(confidences, ground_truth, **kwargs)

    def __str__(self):
        return self.__class__.__name__


class NetcalCalibrationError(ABC):
    def __init__(self, netcal_metric):
        """
        Instance of a netcal metric class, e.g. netcal.metrics.ECE
        """
        self.netcal_metric = netcal_metric

    def _compute(self, confidences: np.ndarray, ground_truth: np.ndarray, **kwargs) \
            -> float:
        return self.netcal_metric.measure(confidences, ground_truth, **kwargs)


class ACE(NetcalCalibrationError):
    """Average Calibration Error. Wraps around netcal's implementation - for further reading refer to netcal's docs."""
    def __init__(self, bins: int = 10):
        super(ACE, self).__init__(netcal.metrics.ACE(bins))


class ECE(NetcalCalibrationError):
    """Expected Calibration Error. Wraps around netcal's implementation - for further reading refer to netcal's docs."""
    def __init__(self, bins: int = 10):
        super().__init__(netcal.metrics.ECE(bins))


class MCE(NetcalCalibrationError):
    """Maximum Calibration Error. Wraps around netcal's implementation - for further reading refer to netcal's docs."""
    def __init__(self, bins: int = 10):
        super().__init__(netcal.metrics.MCE(bins))
