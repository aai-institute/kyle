from typing import Union, Tuple
import numpy as np

from abc import ABC, abstractmethod


class BaseCalibrationError(ABC):

    def __init__(self):
        pass

    @staticmethod
    def input_contains_string(confidences: Union[str, list, np.ndarray], labels: Union[str, list, np.ndarray]) -> bool:
        if type(confidences) is str or type(labels) is str:
            return True
        if isinstance(confidences, list):
            return any(isinstance(elem, str) for elem in confidences)
        if isinstance(labels, list):
            return any(isinstance(elem, str) for elem in labels)
        return False

    def check_input_is_invalid(self, confidences: np.ndarray, labels: np.ndarray) -> Tuple[bool, Union[None, str]]:
        if self.input_contains_string(confidences, labels):
            return True, "Expected input numpy arrays but got str."

        if not np.allclose(np.sum(confidences, axis=1), 1.0, rtol=0.01):
            return True, "Confidences invalid. Probabilities should sum to one."

        return False, None

    @abstractmethod
    def _compute(self, confidences: np.ndarray, ground_truth: np.ndarray, **kwargs) \
            -> Union[float, np.ndarray, ValueError]:
        pass

    def compute(self, confidences: np.ndarray, ground_truth: np.ndarray, **kwargs):
        input_is_invalid, error_message = self.check_input_is_invalid(confidences, ground_truth)
        if input_is_invalid:
            raise ValueError(error_message)
        return self._compute(confidences, ground_truth, **kwargs)
