import numpy as np
import pytest

from kyle.sampling.fake_clf import DirichletFC
from kyle.transformations import ShiftingSimplexAutomorphism


@pytest.fixture(scope="module")
def uncalibrated_samples():
    shifting_vector = np.array([0, 2])
    faker = DirichletFC(
        2, simplex_automorphism=ShiftingSimplexAutomorphism(shifting_vector)
    )
    return faker.get_sample_arrays(1000)


@pytest.fixture(scope="module")
def calibrated_samples():
    faker = DirichletFC(2)
    return faker.get_sample_arrays(1000)
