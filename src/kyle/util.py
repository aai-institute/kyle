from typing import Union

import numpy as np
from sklearn.metrics import accuracy_score


def safe_accuracy_score(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
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


def in_simplex(probabilities: np.ndarray, num_classes=None) -> bool:
    """

    :param probabilities: single vector of probabilities of shape (n_classes,) or multiple
        vectors as array of shape (n_samples, n_classes)
    :param num_classes: if provided, will check whether probability vectors have the correct number of classes
    :return:
    """
    if len(probabilities.shape) == 1:
        probabilities = probabilities[None, :]
    if num_classes is None:
        num_classes = probabilities.shape[1]

    return (
        probabilities.shape[1] == num_classes
        and np.allclose(np.sum(probabilities, axis=1), 1.0, rtol=0.01)
        and (probabilities >= 0).all()
        and (probabilities <= 1).all()
    )


def sample_index(probabilities: np.ndarray) -> Union[int, np.ndarray]:
    """
    Sample indices with the input probabilities. This is essentially a vectorized
    version of np.random.choice

    :param probabilities: single vector of probabilities of shape (n_indices-1,) or multiple
        vectors as array of shape (n_samples, n_indices-1)
    :return: index or array of indices
    """
    rng = np.random.default_rng()
    if len(probabilities.shape) == 1:
        return rng.choice(len(probabilities), p=probabilities)
    elif len(probabilities.shape) == 2:
        # this is a vectorized implementation of np.random.choice with inverse transform sampling
        # see e.g. https://stephens999.github.io/fiveMinuteStats/inverse_transform_sampling.html
        # and https://stackoverflow.com/questions/47722005/vectorizing-numpy-random-choice-for-given-2d-array-of-probabilities-along-an-a
        random_uniform = rng.random(len(probabilities))[:, None]
        return (probabilities.cumsum(axis=1) > random_uniform).argmax(axis=1)
    else:
        raise ValueError(
            f"Unsupported input shape: {probabilities.shape}. Can only be 1 or 2 dimensional."
        )
