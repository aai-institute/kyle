from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score


def safe_accuracy_score(y_true: Sequence, y_pred: Sequence, **kwargs):
    """
    Wrapper around sklearn accuracy store that returns zero for empty sequences of labels

    :param y_true: Ground truth (correct) labels.
    :param y_pred: Predicted labels, as returned by a classifier.
    :param kwargs:
    :return:
    """
    if len(y_true) == len(y_pred) == 0:
        return 0
    return accuracy_score(y_true, y_pred, **kwargs)


def in_simplex(num_classes, x: np.ndarray):
    return len(x) == num_classes and np.isclose(sum(x), 1) and all(x >= 0) and all(x <= 1)