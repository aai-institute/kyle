import pytest
from sklearn import clone

from kyle.calibration.calibration_methods import (
    HistogramBinning,
    TemperatureScaling,
    IsotonicRegression,
    BetaCalibration,
    LogisticCalibration,
)
from kyle.metrics import ECE


@pytest.fixture(scope="module")
def metric():
    return ECE()


@pytest.fixture(scope="module")
def calibration_method():
    return TemperatureScaling()


@pytest.mark.parametrize("calibration_method", [
    HistogramBinning(),
    TemperatureScaling(),
    IsotonicRegression(),
    BetaCalibration(),
    LogisticCalibration(),
])
def test_calibration_methods_clonability(calibration_method):
    clone(calibration_method)


def test_methods_calibrationErrorLessAfterCalibration(
    metric, uncalibrated_samples, calibration_method
):
    ground_truth, confidences = uncalibrated_samples
    error_pre_calibration = metric.compute(confidences, ground_truth)
    calibration_method.fit(confidences, ground_truth)
    calibrated_confidences = calibration_method.get_calibrated_confidences(confidences)
    error_post_calibration = metric.compute(calibrated_confidences, ground_truth)

    assert error_post_calibration <= error_pre_calibration
