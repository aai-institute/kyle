from abc import ABC, abstractmethod
from typing import Callable, Sequence

import numpy as np

from kyle.util import in_simplex


class SimplexAut(ABC):
    """
    Base class for all simplex automorphisms

    :param num_classes: The dimension of the simplex vector, equals 1 + (dimension of the simplex as manifold).
        If provided, will use this for addition I/O checks.
    """

    def __init__(self, num_classes: int = None):
        #  Several transformations can be defined without referring to num_classes, which is why it is optional.
        self.num_classes = num_classes

    def __str__(self):
        return self.__class__.__name__

    @abstractmethod
    def _transform(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: array of shape (n_samples, n_classes)
        :return: transformed array of shape (n_samples, n_classes)
        """
        pass

    def transform(self, x: np.ndarray, check_io=True) -> np.ndarray:
        if len(x.shape) == 1:
            x = x[None, :]
        if check_io and not in_simplex(x, self.num_classes):
            raise ValueError(f"Input has to be from a simplex of suitable dimension")
        x = self._transform(x)
        if check_io and not in_simplex(x, self.num_classes):
            raise ValueError(
                f"Bad implementation: Output has to be from a simplex of suitable dimension"
            )
        return x.squeeze()


class IdentitySimplexAut(SimplexAut):
    def _transform(self, x: np.ndarray) -> np.ndarray:
        return x


class SingleComponentSimplexAut(SimplexAut):
    """
    A simplex automorphism resulting from the application of a map on the unit interval to a
    single component of x and normalizing the result.

    :param component: integer in range [0, num_classes - 1], corresponding to the component on which to apply the mapping
    :param mapping: map from the unit interval [0,1] to itself, should be applicable to arrays
    :param num_classes: The dimension of the simplex vector, equals 1 + (dimension of the simplex as manifold).
        If provided, will use this for addition I/O checks.
    """

    def __init__(
        self,
        component: int,
        mapping: Callable[[np.ndarray], np.ndarray],
        num_classes: int = None,
    ):
        assert (
            0 <= component < num_classes
        ), "Selected component should be in the range [0, num_classes - 1]"
        self.component = component
        self.mapping = mapping
        super().__init__(num_classes=num_classes)

    def _transform(self, x: np.ndarray) -> np.ndarray:
        x = x.copy()
        x[:, self.component] = self.mapping(x[:, self.component])
        return x / x.sum(axis=1)[:, None]


class MaxComponentSimplexAut(SimplexAut):
    """
    A simplex automorphism resulting from the application of a map on the unit interval to a
    the argmax of x and normalizing the remaining components such that the output vector sums to 1.

    :param mapping: map from the unit interval [0,1] to itself, must be applicable to arrays
    :param num_classes: The dimension of the simplex vector, equals 1 + (dimension of the simplex as manifold).
        If provided, will use this for addition I/O checks.
    """

    def __init__(self, mapping: Callable[[np.ndarray], np.ndarray], num_classes=None):
        self.mapping = mapping
        super().__init__(num_classes=num_classes)

    def _transform(self, x: np.ndarray) -> np.ndarray:
        # this transform has a singularity if one component exactly equals one, so we add a minor "noise"
        x = x + 1e-10
        x = x / x.sum(axis=1)[:, None]

        argmax = x.argmax(axis=1)
        old_values = np.choose(argmax, x.T)
        new_values = self.mapping(old_values)
        # the result must sum to 1, so we will rescale the remaining entries of the confidence vectors
        remaining_comps_normalization = (1 - new_values) / (1 - old_values)
        new_values_compensated_for_norm = new_values / remaining_comps_normalization
        np.put_along_axis(
            x, argmax[:, None], new_values_compensated_for_norm[:, None], axis=1
        )
        return x * remaining_comps_normalization[:, None]


class PowerLawSimplexAut(SimplexAut):
    """
    An automorphism resulting from taking elementwise powers of the inputs with fixed exponents
    and normalizing the result.

    |
    | *Intuition*:

    If exponents[j] < exponents[i], then the output will be more shifted towards the j-th direction
    than the i-th. If all exponents are equal to some number s, then s>1 means a shift towards the boundary
    of the simplex whereas 0<s<1 means a shift towards the center and s < 0 results in an "antipodal shift".

    :param exponents: sequence of length num_classes
    """

    def __init__(self, exponents: Sequence[float]):
        self.exponents = np.array(exponents)
        super().__init__(len(exponents))

    def _transform(self, x: np.ndarray) -> np.ndarray:
        x = np.float_power(x, self.exponents)
        return x / x.sum(axis=1)[:, None]


class RestrictedPowerSimplexAut(SimplexAut):
    """
    Maybe a bad idea, feels unnatural
    """

    def __init__(self, exponents: np.ndarray):
        """

        :param exponents: numpy array of shape (num_classes - 1, )
        """
        if not np.all(exponents >= 1):
            raise ValueError("Only exponents >= 1 are permitted")
        self.exponents = exponents[None, :]
        super().__init__(len(exponents) + 1)

    def _transform(self, x: np.ndarray) -> np.ndarray:
        x = x.copy()
        x[:, :-1] = np.float_power(x[:, :-1], self.exponents)
        x[:, -1] = 1 - x[:, :-1].sum(axis=1)
        return x / x.sum(axis=1)[:, None]
