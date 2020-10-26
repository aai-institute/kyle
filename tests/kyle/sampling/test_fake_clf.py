from kyle.sampling.fake_clf import SufficientlyConfidentFC, DirichletFC


def test_SufficientlyConfidentFC():
    n_classes, n_samples = 3, 5
    faker = SufficientlyConfidentFC(n_classes)
    ground_truth, confidences = faker.get_sample()
    ground_truth_array, class_proba_array = faker.get_sample_arrays(5)
    assert ground_truth in range(n_classes)
    assert confidences.shape == (n_classes,)
    assert ground_truth_array.shape == (n_samples, )
    assert class_proba_array.shape == (n_samples, n_classes)


def test_DirichletFC():
    faker = DirichletFC(3)
    ground_truth, class_proba = faker.get_sample()
    assert ground_truth in [0, 1, 2]
    assert len(class_proba) == 3
