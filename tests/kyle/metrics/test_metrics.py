import pytest

from kyle.metrics import ECE, MCE, ACE
from kyle.sampling.fake_clf import DirichletFC


@pytest.fixture(scope="module")
def metrics():
    criteria = [ECE(), MCE(), ACE()]
    return criteria


@pytest.fixture(scope="module")
def samples():
    faker = DirichletFC(2)
    return faker.get_sample_arrays(1000)


def test_metrics_calibratedConfidencesHaveZeroError(metrics, samples):
    for criterion in metrics:
        ground_truth, confidences = samples
        epsilon = 0.1
        assert criterion.compute(confidences, ground_truth) <= epsilon
