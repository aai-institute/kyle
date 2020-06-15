from collections import defaultdict
from typing import Dict, Any, Sequence, Iterable, TypeVar, Optional, List

T = TypeVar("T")


def iter_param_combinations(hyper_param_values: Dict[Any, Sequence]) -> Iterable[Dict[str, Any]]:
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
            yield params
        else:
            param_name, param_values = pairs[i]
            for param_value in param_values:
                params[param_name] = param_value
                yield from _iter_recursive_param_combinations(pairs, i + 1, params)

    return _iter_recursive_param_combinations(input_pairs, 0, {})


def getFirstDuplicate(seq: Sequence[T]) -> Optional[T]:
    """
    Returns the first duplicate in a sequence or None

    :param seq: a sequence of hashable elements
    :return:
    """
    setOfElems = set()
    for elem in seq:
        if elem in setOfElems:
            return elem
        setOfElems.add(elem)


def updateDictByListingValuesAndNormalize(dicts: List[dict]) -> dict:
    """
    merge a list of dictionaries together into a single dictionary. For every key generate a list of all values in each dictionary.
    If this list contains only the same value, normalize the list into this value.

    e.g.

    ``[{"key1": 1, "key2": 2}, {"key1": 1, "key2": 3}]`` -> ``{"key1": 1, "key": [2, 3]}``

    :param dicts: list of ditionaries
    :return: dictionary
    """
    def _normalizeListValues(input_dict: Dict[List]):
        return {key: values[0] if all(x == values[0] for x in values) else values for key, values in input_dict.items()}

    merged = defaultdict(list)
    for d in dicts:
        for k, v in d.items():
            merged[k].append(v)
    merged = _normalizeListValues(merged)
    return merged
