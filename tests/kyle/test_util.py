import numpy as np

from kyle.util import in_simplex


def test_in_simplex_negativeEntriesForbidden():
    assert not in_simplex(np.array([0.5, -0.5]))


def test_in_simplex_larger1Forbidden():
    assert not in_simplex(np.array([0, 2]))


def test_in_simplex_sumNot1Forbidden():
    assert not in_simplex(np.array([0.4, 0.7]))
    assert not in_simplex(np.array([0.1, 0.1]))
    assert not in_simplex(np.random.default_rng().random((5, 3)))


def test_in_simplex_wrongSizeForbidden():
    assert not in_simplex(np.array([1]), num_classes=2)
    assert not in_simplex(np.array([1, 0, 0]), num_classes=2)
    assert not in_simplex(np.random.default_rng().random((5, 3)), num_classes=2)


def test_in_simplex_correctInputIsCorrect():
    assert in_simplex(np.array([0.5, 0.5]), num_classes=2)
    x = np.random.default_rng().random(5)
    assert in_simplex(x / x.sum())


def test_in_simplex_correct2DInputIsCorrect():
    x = np.random.default_rng().random((5, 3))
    row_sums = x.sum(axis=1)
    x = x / row_sums[:, np.newaxis]
    assert in_simplex(x)
    assert in_simplex(x, num_classes=3)
