from kale.sampling.fake_clf import DirichletFC


def sample_confs_from_dirichlet(n_classes: int = 2, n_samples: int = 1000):
    faker = DirichletFC(n_classes)
    return faker.get_sample_arrays(n_samples)
