from kale.sampling import FakeClassifier


def test_FakeClassifier():
    faker = FakeClassifier(3)
    ground_truth, class_proba = faker.get_sample()
    assert ground_truth in [0, 1, 2]
    assert len(class_proba) == 3
