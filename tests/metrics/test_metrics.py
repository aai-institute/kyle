import pytest

from kale.sampling.fake_clf import DirichletFC
from metrics import ECE


# TODO:
#   Generalize test suite to all measures (Add common_metric_tests file and import functions from there).
#   Add sampling method to a new file test_utils.
#   Do same for all other metrics as ECE

@pytest.fixture()
def criterion():
    criterion = ECE()
    return criterion


def test_string_input_validation(criterion):
    with pytest.raises(ValueError):
        assert criterion.measure(["8"], ["10"])
        assert criterion.measure("8", "10")
        assert criterion.measure(["8", [1, 2]])


def test_confidences_sum_to_one(criterion):
    n_classes, n_samples = 2, 1000
    faker = DirichletFC(n_classes)
    ground_truth, confidences = faker.get_sample_arrays(n_samples)
    confidences += 0.5
    with pytest.raises(ValueError):
        assert criterion.measure(confidences, ground_truth)


def test_with_calibrated_confidences(criterion):
    n_classes, n_samples = 2, 1000
    faker = DirichletFC(n_classes)
    ground_truth, confidences = faker.get_sample_arrays(n_samples)
    epsilon = 0.1
    assert criterion.measure(confidences, ground_truth) <= epsilon
