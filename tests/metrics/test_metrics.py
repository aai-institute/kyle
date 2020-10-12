import pytest

from metrics import ECE, MCE, ACE
from tests.metrics.utils import sample_confs_from_dirichlet


@pytest.fixture()
def criteria():
    criteria = [ECE(), MCE(), ACE()]
    return criteria


def test_string_input_validation(criteria):
    for criterion in criteria:
        with pytest.raises(ValueError):
            assert criterion.measure(["8"], ["10"])
            assert criterion.measure("8", "10")
            assert criterion.measure(["8", [1, 2]])


def test_confidences_sum_to_one(criteria):
    for criterion in criteria:
        ground_truth, confidences = sample_confs_from_dirichlet(n_classes=2, n_samples=1000)
        confidences += 0.5
        with pytest.raises(ValueError):
            assert criterion.measure(confidences, ground_truth)


def test_with_calibrated_confidences(criteria):
    for criterion in criteria:
        ground_truth, confidences = sample_confs_from_dirichlet(n_classes=2, n_samples=1000)
        epsilon = 0.1
        assert criterion.measure(confidences, ground_truth) <= epsilon
