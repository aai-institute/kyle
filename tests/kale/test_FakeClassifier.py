from kale.sampling import FakeClassifier


def test_FakeClassifier():
    n_classes, n_samples = 3, 5
    faker = FakeClassifier(n_classes)
    ground_truth, class_proba = faker.get_sample()
    ground_truth_array, class_proba_array = faker.get_sample_arrays(5)
    assert ground_truth in range(n_classes)
    assert class_proba.shape == (n_classes,)
    assert ground_truth_array.shape == (n_samples, )
    assert class_proba_array.shape == (n_samples, n_classes)
