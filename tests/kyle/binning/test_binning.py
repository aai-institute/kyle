from kyle.evaluation.discrete import HomogeneousBinning
import numpy as np


def test_homogeneous():
    homogenous_binning = HomogeneousBinning(
        confidences=np.array([0.1, 0.3, 0.5, 0.7, 0.9]), bins=5
    )
    result = homogenous_binning.set_bins()
    expected_result = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    assert (result == expected_result).all()
