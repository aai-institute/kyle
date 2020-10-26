import numpy as np

from kyle.util import in_simplex


def test_in_simplex_negativeEntriesForbidden():
    assert not in_simplex(2, np.array([0.5, -0.5]))


def test_in_simplex_larger1Forbidden():
    assert not in_simplex(2, np.array([0, 2]))


def test_in_simplex_sumNot1Forbidden():
    assert not in_simplex(2, np.array([0.4, 0.7]))
    assert not in_simplex(2, np.array([0.1, 0.1]))


def test_in_simplex_wrongSizeForbidden():
    assert not in_simplex(2, np.array([1]))
    assert not in_simplex(2, np.array([1, 0, 0]))


def test_in_simplex_correctInputIsCorrect():
    assert in_simplex(2, np.array([0.5, 0.5]))
    x = np.random.default_rng().random(5)
    assert in_simplex(5, x/x.sum())
