import numpy as np
from scipy.integrate import quad
from scipy.stats import dirichlet

from kyle.integrals import (
    dirichlet_exp_value,
    simplex_integral_fixed_comp,
    simplex_integral_fixed_max,
)
from kyle.sampling.fake_clf import DirichletFC
from kyle.transformations import SimplexAut


def _prob_correct_prediction(conf: np.ndarray, simplex_aut: SimplexAut):
    conf = conf.squeeze()
    gt_probabilities = simplex_aut.transform(conf, check_io=False)
    return gt_probabilities[np.argmax(conf)]


def _prob_class(conf: np.ndarray, simplex_aut: SimplexAut, selected_class: int):
    conf = conf.squeeze()
    gt_probabilities = simplex_aut.transform(conf, check_io=False)
    return gt_probabilities[selected_class]


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


def compute_ECE(dirichlet_fc: DirichletFC, conditioned="full", **kwargs):
    """
    Computes theoretical ECE of dirichlet_fc Fake Classifier conditioned on full confidence vector, conditioned on the
    confidence in prediction or conditioned on each class confidence separately (see [1]_ for further details)

    :param dirichlet_fc: Dirichlet fake classifier to calculate ECE for
    :param conditioned: Quantity to condition ECE on
    :param kwargs: passed to integrator function
    :return: * If conditioned on full confidence vector returns: result, abserr, (further scipy.nquad output)
             * If conditioned on the confidence in prediction returns: result, abserr, (further scipy.quad output)
             * If conditioned on each class separately returns: List of num_classes+1 entries. First entry contains
               average of all "i-class ECEs". Subsequent entries contain results for each "i-class ECE"
               separately: result, abserr, (further scipy.quad output)

    References
    ----------
    .. [1] Kull, M., Perello-Nieto, M., Kängsepp, M., Filho, T. S., Song, H., & Flach, P. (2019). Beyond temperature
        scaling: Obtaining well-calibrated multiclass probabilities with Dirichlet calibration.
    """

    if conditioned == "full":
        return _compute_ECE_full(dirichlet_fc, **kwargs)
    elif conditioned == "confidence":
        return _compute_ECE_conf(dirichlet_fc, **kwargs)
    elif conditioned == "class":
        return _compute_ECE_class(dirichlet_fc, **kwargs)
    else:
        raise ValueError("ECE has to be one of fully, confidence or class conditioned")


def _compute_ECE_full(dirichlet_fc: DirichletFC, **kwargs):
    def integrand(*parametrization):
        conf = _probability_vector(*parametrization)
        return np.abs(
            np.max(conf)
            - _prob_correct_prediction(conf, dirichlet_fc.simplex_automorphism)
        )

    return dirichlet_exp_value(integrand, dirichlet_fc.alpha, **kwargs)


def _compute_ECE_conf(dirichlet_fc: DirichletFC, **kwargs):
    # Need higher precision for accurate result due to nesting of two quad/nquad calls
    # Sets higher precision if precision not already set in **kwargs
    opts = {"epsabs": 1e-4}
    opts.update(kwargs.pop("opts", {}))
    kwargs.update({"opts": opts})

    num_classes = len(dirichlet_fc.alpha)

    def p_c(*parametrization):
        return dirichlet.pdf(parametrization, dirichlet_fc.alpha)

    def p_y_c(*parametrization):
        conf = _probability_vector(*parametrization)
        return _prob_correct_prediction(conf, dirichlet_fc.simplex_automorphism) * p_c(
            *parametrization
        )

    def integrand(max_conf):
        int_p_c = simplex_integral_fixed_max(p_c, num_classes, max_conf, **kwargs)[0]
        int_p_y_c = simplex_integral_fixed_max(p_y_c, num_classes, max_conf, **kwargs)[
            0
        ]
        return np.abs(int_p_y_c / int_p_c - max_conf) * int_p_c

    # At exactly 1/num_classes or 1 get 0/0
    boundary_offset = 1e-2

    return quad(
        integrand,
        1 / num_classes + boundary_offset,
        1 - boundary_offset,
        epsabs=opts["epsabs"],
    )


def _compute_ECE_class(dirichlet_fc: DirichletFC, **kwargs):
    # Need higher precision for accurate result due to nesting of two quad/nquad calls
    # Sets higher precision if precision not already set in **kwargs
    opts = {"epsabs": 1e-4}
    opts.update(kwargs.pop("opts", {}))
    kwargs.update({"opts": opts})

    num_classes = len(dirichlet_fc.alpha)

    integral_results = []

    for i in range(num_classes):

        def p_c(*parametrization):
            return dirichlet.pdf(parametrization, dirichlet_fc.alpha)

        def p_y_c(*parametrization):
            conf = _probability_vector(*parametrization)
            return _prob_class(conf, dirichlet_fc.simplex_automorphism, i) * p_c(
                *parametrization
            )

        def integrand(comp_conf):
            int_p_c = simplex_integral_fixed_comp(
                p_c, num_classes, i, comp_conf, **kwargs
            )[0]
            int_p_y_c = simplex_integral_fixed_comp(
                p_y_c, num_classes, i, comp_conf, **kwargs
            )[0]
            return np.abs(int_p_y_c / int_p_c - comp_conf) * int_p_c

        # At exactly 0 or 1 get 0/0
        boundary_offset = 1e-2

        result = quad(
            integrand,
            1 / num_classes + boundary_offset,
            1 - boundary_offset,
            epsabs=opts["epsabs"],
        )

        integral_results.append(result)

    integral_results.insert(0, sum(S[0] for S in integral_results) / num_classes)

    return integral_results
