import pytest

from kyle.metrics import ACE, ECE, MCE


@pytest.fixture(scope="module")
def metrics():
    criteria = [ECE(), MCE(), ACE()]
    return criteria


def test_metrics_calibratedConfidencesHaveZeroError(metrics, calibrated_samples):
    ground_truth, confidences = calibrated_samples
    for criterion in metrics:
        epsilon = 0.1
        assert criterion.compute(confidences, ground_truth) <= epsilon


def test_metrics_uncalibratedConfidencesHaveNonZeroError(metrics, uncalibrated_samples):
    ground_truth, confidences = uncalibrated_samples
    for criterion in metrics:
        epsilon = 0.1
        assert criterion.compute(confidences, ground_truth) > epsilon
