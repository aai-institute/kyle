from typing import Union, Tuple
import numpy as np

from abc import ABC, abstractmethod


class BaseCalibrationError(ABC):

    def __init__(self):
        pass

    @staticmethod
    def check_confidences_sum_to_one(confidences: np.ndarray) -> bool:
        max_probability_threshold = 1.0001
        return (np.sum(confidences, axis=1) < max_probability_threshold).all()

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
        if not self.check_confidences_sum_to_one(confidences=confidences):
            return True, "Confidences invalid. Probabilities should sum to one."
        return False, None

    @abstractmethod
    def measure(self, confidences: np.ndarray, ground_truth: np.ndarray):
        pass
