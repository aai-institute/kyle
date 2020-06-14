from typing import Sequence

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