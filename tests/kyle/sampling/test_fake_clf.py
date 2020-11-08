from kyle.sampling.fake_clf import DirichletFC
from kyle.util import in_simplex


def test_DirichletFC_basics():
    faker = DirichletFC(3)
    ground_truth, class_proba = faker.get_sample_arrays(10)
    assert ground_truth.shape == (10,)
    assert class_proba.shape == (10, 3)
    assert ground_truth[0] in [0, 1, 2]
    assert in_simplex(class_proba)
