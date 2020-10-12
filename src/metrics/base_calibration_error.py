import numpy as np
from typing import Union
from abc import ABC, abstractmethod


class BaseCalibrationError(ABC):

    def __init__(self):
        pass

    @staticmethod
    def check_confidences_sum_to_one(confidences: np.ndarray) -> bool:
        max_probability_threshold = 1.0001
        return (np.sum(confidences, axis=1) < max_probability_threshold).all()

    def validate_input(self, confidences: np.ndarray, ground_truth: np.ndarray) -> Union[None, str]:
        if not self.check_confidences_sum_to_one(confidences=confidences):
            return "Confidences invalid. Probabilities should sum to one."
        return None

    @abstractmethod
    def measure(self, confidences: np.ndarray, ground_truth: np.ndarray):
        pass
