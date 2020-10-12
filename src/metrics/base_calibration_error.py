import numpy as np
from abc import ABC, abstractmethod


class BaseCalibrationError(ABC):

    def __init__(self):
        pass

    def validate_input(self):
        pass

    @abstractmethod
    def measure(self, confidences: np.ndarray, ground_truth: np.ndarray):
        pass
