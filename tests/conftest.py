import numpy as np
import pytest

from kyle.sampling.fake_clf import DirichletFC
from kyle.transformations import PowerLawSimplexAut


@pytest.fixture(scope="module")
def uncalibrated_samples():
    faker = DirichletFC(2, simplex_automorphism=PowerLawSimplexAut(np.array([30, 20])))
    return faker.get_sample_arrays(1000)


@pytest.fixture(scope="module")
def calibrated_samples():
    faker = DirichletFC(2)
    return faker.get_sample_arrays(1000)
