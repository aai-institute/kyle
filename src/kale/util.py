from typing import Dict, Sequence, Iterable, TypeVar, Optional

import numpy as np
# from sklearn.metrics import accuracy_score - see comment below, next to our implementation of accuracy_score

T = TypeVar("T")
S = TypeVar("S")


def iter_param_combinations(hyper_param_values: Dict[S, Sequence[T]]) -> Iterable[Dict[S, T]]:
    """
    Create all possible combinations of values from a dictionary of possible parameter values

    :param hyper_param_values: a mapping from parameter names to lists of possible values
    :return: an iterator of dictionaries mapping each parameter name to one of the values
    """
    input_pairs = list(hyper_param_values.items())

    def _iter_recursive_param_combinations(pairs, i, params: Dict):
        """
        Recursive function to create all possible combinations from a list of key-array entries

        :param pairs: a dictionary of parameter names and their corresponding values
        :param i: the recursive step
        :param params: a dictionary to which iteration results will be aggregated
        """
        if i == len(pairs):  # there are len(pairs) + 1 recursive steps in total
            yield dict(params)
        else:
            param_name, param_values = pairs[i]
            for param_value in param_values:
                params[param_name] = param_value
                yield from _iter_recursive_param_combinations(pairs, i + 1, params)

    return _iter_recursive_param_combinations(input_pairs, 0, {})


def get_first_duplicate(seq: Sequence[T]) -> Optional[T]:
    """
    Returns the first duplicate in a sequence or None

    :param seq: a sequence of hashable elements
    :return:
    """
    set_of_elems = set()
    for elem in seq:
        if elem in set_of_elems:
            return elem
        set_of_elems.add(elem)


# NOTE: this should be adjusted in the pyodide-packaging independent version of kale
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
    try:
        from sklearn.metrics import accuracy_score
        return accuracy_score(y_true, y_pred, **kwargs)
    except ImportError:
        if kwargs is not None:
            raise NotImplementedError("kwargs are currently not supported in the pyodide-packaged version of kale")
        return custom_accuracy_score(y_true, y_pred)


# IMPORTANT: the only reason for this method is to not have sklearn as dependency in the pyodide
# package (and thereby significantly reduce its size).
# When and if we package kale as a separate library, independent of the pyodide packages and the game,
# this method will be removed in favor of including sklearn as dependency
def custom_accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.shape != y_pred.shape:
        raise Exception("unequal shapes")
    if len(y_true.shape) != 1:
        raise NotImplementedError("Only implemented for 1-dim. arrays in the pyodide-packaged version of kale")
    return (y_true == y_pred).mean()



def in_simplex(num_classes, x: np.ndarray) -> bool:
    return len(x) == num_classes and np.isclose(sum(x), 1) and all(x >= 0) and all(x <= 1)
