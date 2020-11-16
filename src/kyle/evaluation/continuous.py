import numpy as np

from kyle.integrals import dirichlet_exp_value
from kyle.sampling.fake_clf import DirichletFC
from kyle.transformations import SimplexAut


def _prob_correct_prediction(conf: np.ndarray, simplex_aut: SimplexAut):
    conf = conf.squeeze()
    gt_probabilities = simplex_aut.transform(conf, check_io=False)
    return gt_probabilities[np.argmax(conf)]


def _probability_vector(*parametrization: float):
    return np.array(list(parametrization) + [1 - np.sum(parametrization)])


def compute_accuracy(dirichlet_fc: DirichletFC, **kwargs):
    def integrand(*parametrization):
        conf = _probability_vector(*parametrization)
        return _prob_correct_prediction(conf, dirichlet_fc.simplex_automorphism)

    return dirichlet_exp_value(integrand, dirichlet_fc.alpha, **kwargs)


def compute_expected_max(dirichlet_fc: DirichletFC, **kwargs):
    def integrand(*parametrization):
        conf = _probability_vector(*parametrization)
        return np.max(conf)

    return dirichlet_exp_value(integrand, dirichlet_fc.alpha, **kwargs)


def compute_ECE(dirichlet_fc: DirichletFC, **kwargs):
    def integrand(*parametrization):
        conf = _probability_vector(*parametrization)
        return np.abs(
            np.max(conf)
            - _prob_correct_prediction(conf, dirichlet_fc.simplex_automorphism)
        )

    return dirichlet_exp_value(integrand, dirichlet_fc.alpha, **kwargs)
