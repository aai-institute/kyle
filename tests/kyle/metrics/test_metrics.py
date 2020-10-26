import numpy as np
import pytest

from kyle.metrics import ECE, MCE, ACE
from kyle.sampling.fake_clf import DirichletFC


@pytest.fixture(scope="module")
def metrics():
    criteria = [ECE(), MCE(), ACE()]
    return criteria


def sample_confs_from_dirichlet(n_classes: int = 2, n_samples: int = 1000) -> np.ndarray:
    faker = DirichletFC(n_classes)
    return faker.get_sample_arrays(n_samples)


def test_with_calibrated_confidences(metrics):
    for criterion in metrics:
        ground_truth, confidences = sample_confs_from_dirichlet(n_classes=2, n_samples=1000)
        epsilon = 0.1
        assert criterion.compute(confidences, ground_truth) <= epsilon
