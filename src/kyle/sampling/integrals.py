from typing import Callable

import numpy as np
from scipy.integrate import nquad
from scipy.stats import dirichlet

from kyle.sampling.fake_clf import DirichletFC
from kyle.transformations import SimplexAutomorphism


def simplex_integral(f: Callable, num_classes: int, **kwargs):
    if num_classes < 2:
        raise ValueError("need at least two classes")

    def nested_variable_boundary(*previous_variables: float):
        """
        Any inner variable for the simplex integral
        goes from zero to 1 - sum(all previous variables).
        """
        if len(previous_variables) == 0:
            return [0, 1]
        return [0, 1 - sum(previous_variables)]

    simplex_boundary = [nested_variable_boundary] * (num_classes - 1)

    if "opts" not in kwargs:
        # we typically don't need high precision
        opts = {"epsabs": 1e-2}
    else:
        kwargs = kwargs.copy()
        opts = kwargs.pop("opts")
    return nquad(f, simplex_boundary, opts=opts, **kwargs)


def dirichlet_exp_value(f: Callable, alpha: np.ndarray, **kwargs):
    num_classes = len(alpha)
    return simplex_integral(
        lambda *args: f(*args) * dirichlet.pdf(args, alpha), num_classes, **kwargs
    )


def get_argmax_region_char_function(selected_class: int) -> Callable[[...], float]:
    """
    Returns the char. function for the area in which the selected class is the argmax of the input args.
    The returned function takes a variable number of floats as input. They represent the first N-1 independent
    entries of an element of a simplex in N-dimensional space (N classes).
    """

    def char_function(*args: float):
        if len(args) < 2:
            raise ValueError("need at least two classes")
        if not 0 <= selected_class <= len(args):
            raise IndexError(
                f"selected_class {selected_class} out of bound for input of length {len(args)}"
            )
        confidences = list(args) + [1 - sum(args)]
        class_confidence = confidences[selected_class]
        return float(class_confidence == np.max(confidences))

    return char_function


def _prob_correct_prediction(conf: np.ndarray, simplex_aut: SimplexAutomorphism):
    conf = conf.squeeze()
    gt_probabilities = simplex_aut.transform(conf, check_io=False)
    return gt_probabilities[np.argmax(conf)]


def compute_accuracy(dirichlet_fc: DirichletFC):
    def integrand(conf):
        return _prob_correct_prediction(conf, dirichlet_fc.simplex_automorphism)

    return dirichlet_exp_value(integrand, dirichlet_fc.alpha)


def compute_expected_max(dirichlet_fc: DirichletFC):
    return dirichlet_exp_value(lambda x: np.max(x), dirichlet_fc.alpha)


def compute_ECE(dirichlet_fc: DirichletFC):
    def integrand(conf):
        return np.max(conf) - _prob_correct_prediction(
            conf, dirichlet_fc.simplex_automorphism
        )

    return dirichlet_exp_value(integrand, dirichlet_fc.alpha)
