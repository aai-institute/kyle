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


def simplex_integral(
    f: Callable, num_classes: int, boundary_offset=1e-10, coord_sum: float = 1, **kwargs
):
    """
    Performs an integral over num_classes-1 dimensional simplex using scipy

    :param f: function to integrate over the simplex. Should accept num_classes-1 variables
    :param num_classes: equals dimension of the simplex + 1
    :param boundary_offset: can be used to prevent numerical errors due to singularities at the simplex' boundary
    :param coord_sum: sets sum of coordinates of simplex. For standard simplex sum(x1,x2,...) = 1. Mainly useful for
        simplex_integral_fixed_max
    :param kwargs: will be passed to scipy.integrate.nquad
    :return:
    """
    if num_classes < 2:
        raise ValueError("need at least two classes")

    def nested_variable_boundary(*previous_variables: float):
        """
        Any variable for the simplex integral goes from zero to coord_sum (usually 1) - sum(all previous variables).
        See docu of nquad for more details on boundaries
        """
        return [
            0 + boundary_offset,
            coord_sum - sum(previous_variables) - boundary_offset,
        ]

    simplex_boundary = [nested_variable_boundary] * (num_classes - 1)
    # we typically don't need higher precision
    opts = {"epsabs": 1e-2}
    opts.update(kwargs.pop("opts", {}))
    return nquad(f, simplex_boundary, opts=opts, **kwargs)


def simplex_integral_fixed_comp(
    f: Callable, num_classes: int, selected_class: int, x_comp: float, **kwargs
):
    """
    Performs an integral over the subset of a num_classes-1 dimensional simplex defined by the selected_class component
    of the confidence vector having a fixed value of x_comp, i.e. marginalises out all other classes.

    Computing this involves integrating over a num_classes-2 dimensional non-unit simplex with coord_sum set to 1-x_comp
    and with the selected_class argument of f being set to x_comp

    :param f: function to integrate over the subset of the simplex. Should accept num_classes-1 variables
    :param num_classes: equals dimension of the simplex + 1
    :param selected_class: selected confidence vector component [0, num_classes-1]
    :param x_comp: fixed value of the selected vector component
    :param kwargs: passed to simplex_integral
    :return:
    """

    if not (0 <= x_comp <= 1):
        raise ValueError("Confidences have to lie in range (0,1)")

    if selected_class == num_classes - 1:

        def constrained_integrand(*args: float):
            constrained_args = [1 - x_comp - sum(args[0:]), *args[0:]]
            return f(*constrained_args)

    else:

        def constrained_integrand(*args: float):
            constrained_args = [*args[0:selected_class], x_comp, *args[selected_class:]]
            return f(*constrained_args)

    return simplex_integral(
        constrained_integrand, num_classes - 1, coord_sum=1 - x_comp, **kwargs
    )


def simplex_integral_fixed_max(f: Callable, num_classes: int, x_max: float, **kwargs):
    """
    Performs an integral over the subset of a num_classes-1 dimensional simplex defined by the largest
    coordinate/confidence having a fixed value of x_max, i.e. marginalises over all possible confidence vectors with
    maximum confidence of x_max.

    Computing this integral involves computing the sum of num_classes integrals each over a num_classes-2 dimensional
    simplex. For x_max > 0.5 the integrals are 'true' simplex integrals. For x_max < 0.5 the boundaries become complex
    and non-simplex like. The integrals can then be extended to full simplex integrals using an appropiate indicator
    function, ``get_argmax_region_char_function``.

    :param f: function to integrate over the subset of the simplex. Should accept num_classes-1 variables
    :param num_classes: equals dimension of the simplex + 1
    :param x_max: fixed value of largest coordinate value. defines subset of simplex
    :param kwargs: passed to simplex_integral_fixed_comp
    :return:
    """

    if not (1 / num_classes < x_max < 1):
        return 0, 0

    # For small x_max higher precision is required for accurate results as over large ingtegration range integrand is 0
    # Sets higher precision if precision not already set in **kwargs
    if x_max < 1 / 2:
        opts = {"epsabs": 1e-4}
        opts.update(kwargs.pop("opts", {}))
        kwargs.update({"opts": opts})

    integral_result = (0, 0)

    for i in range(num_classes):

        argmax_char_func = get_argmax_region_char_function(i)

        constrained_integral = simplex_integral_fixed_comp(
            lambda *args: argmax_char_func(*args) * f(*args),
            num_classes,
            i,
            x_max,
            **kwargs,
        )
        integral_result = tuple(
            sum(p) for p in zip(integral_result, constrained_integral)
        )

    return integral_result


def dirichlet_exp_value(f: Callable, alpha: Sequence[float], **kwargs):
    """
    Computes expectation value of f over num_classes-1 dimensional simplex using scipy. Note scipy.dirichlet.pdf for
    n classes accepts n-1 entries as sum(x_n) = 1.

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
        if len(args) < 1:
            raise ValueError("need at least two classes/one input")
        if not 0 <= selected_class <= len(args):
            raise IndexError(
                f"selected_class {selected_class} out of bound for input of length {len(args)}"
            )
        probabilities = list(args) + [1 - sum(args)]
        class_confidence = probabilities[selected_class]
        return float(class_confidence == max(probabilities))

    return char_function


log = logging.getLogger(__name__)
