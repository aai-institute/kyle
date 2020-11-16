import logging
from typing import Callable, Protocol, Sequence

from scipy.integrate import nquad
from scipy.stats import dirichlet


# this is the currently supported way to annotate callables with *args of a certain type,
# see https://mypy.readthedocs.io/en/latest/protocols.html#callback-protocols
# hopefully at some point the pycharm type checker will learn to recognize those.
# I opened an issue for JetBrains: https://youtrack.jetbrains.com/issue/PY-45438
class Integrand(Protocol):
    def __call__(self, *parameters: float) -> float:
        ...


def simplex_integral(f: Callable, num_classes: int, boundary_offset=1e-10, **kwargs):
    """
    Performs an integral over num_classes-1 dimensional simplex using scipy

    :param f: function to integrate over the simplex. Should accept num_classes-1 variables
    :param num_classes: equals dimension of the simplex + 1
    :param boundary_offset: can be used to prevent numerical errors due to singularities at the simplex' boundary
    :param kwargs: will be passed to scipy.integrate.nquad
    :return:
    """
    if num_classes < 2:
        raise ValueError("need at least two classes")

    def nested_variable_boundary(*previous_variables: float):
        """
        Any variable for the simplex integral goes from zero to 1 - sum(all previous variables).
        See docu of nquad for more details on boundaries
        """
        return [0 + boundary_offset, 1 - sum(previous_variables) - boundary_offset]

    simplex_boundary = [nested_variable_boundary] * (num_classes - 1)
    # we typically don't need higher precision
    opts = {"epsabs": 1e-2}
    opts.update(kwargs.pop("opts", {}))
    return nquad(f, simplex_boundary, opts=opts, **kwargs)


def dirichlet_exp_value(f: Callable, alpha: Sequence[float], **kwargs):
    """
    Computes expectation value of f over num_classes-1 dimensional simplex using scipy

    :param f:
    :param alpha: the parameters of the dirichlet distribution, one for each class
    :param kwargs: passed to simplex_integral
    :return:
    """
    num_classes = len(alpha)
    return simplex_integral(
        lambda *args: f(*args) * dirichlet.pdf(args, alpha), num_classes, **kwargs
    )


def get_argmax_region_char_function(selected_class: int) -> Integrand:
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
        probabilities = list(args) + [1 - sum(args)]
        class_confidence = probabilities[selected_class]
        return float(class_confidence == max(probabilities))

    return char_function


log = logging.getLogger(__name__)
