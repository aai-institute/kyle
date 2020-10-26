import pytest
import numpy as np

from kale.metrics import ECE, MCE, ACE
from kale.sampling.fake_clf import DirichletFC


@pytest.fixture(scope="module")
def criteria():
    criteria = [ECE(), MCE(), ACE()]
    return criteria


def sample_confs_from_dirichlet(n_classes: int = 2, n_samples: int = 1000) -> np.ndarray:
    faker = DirichletFC(n_classes)
    return faker.get_sample_arrays(n_samples)


def test_confidences_sum_to_one(criteria: list):
    for criterion in criteria:
        ground_truth, confidences = sample_confs_from_dirichlet(n_classes=2, n_samples=1000)
        confidences += 0.5
        with pytest.raises(ValueError):
            assert criterion.compute(confidences, ground_truth)


def test_with_calibrated_confidences(criteria: list):
    for criterion in criteria:
        ground_truth, confidences = sample_confs_from_dirichlet(n_classes=2, n_samples=1000)
        epsilon = 0.1
        assert criterion.compute(confidences, ground_truth) <= epsilon
