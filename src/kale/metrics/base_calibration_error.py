from typing import Union, Tuple
import numpy as np

from abc import ABC, abstractmethod


class BaseCalibrationError(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def _compute(self, confidences: np.ndarray, ground_truth: np.ndarray, **kwargs) \
            -> Union[float, np.ndarray, ValueError]:
        pass

    def compute(self, confidences: np.ndarray, ground_truth: np.ndarray, **kwargs):
        if not np.allclose(np.sum(confidences, axis=1), 1.0, rtol=0.01):
            raise ValueError("Confidences invalid. Probabilities should sum to one.")
        return self._compute(confidences, ground_truth, **kwargs)
