import pytest

from kale.util import custom_accuracy_score
import numpy as np


@pytest.mark.parametrize("y_true,y_pred,expected", [
    ([0], [0], 1.0),
    ([0], [1], 0.0),
    ([0, 1], [0, 0], 0.5)
])
def test_custom_accuracy_store(y_true, y_pred, expected):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    assert custom_accuracy_score(y_true, y_pred) == expected
    with pytest.raises(Exception):
        custom_accuracy_score(np.ones([1, 2]), np.ones(1))
