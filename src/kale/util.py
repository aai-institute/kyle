from typing import Dict, Sequence, Iterable, TypeVar, Optional

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
