from abc import ABC, abstractmethod


class BaseCalibration:
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def adjust_confidences(self, confidences):
        pass
