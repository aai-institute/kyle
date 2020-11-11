from abc import abstractmethod, ABC
from typing import Callable

import numpy as np

from kyle.util import in_simplex


class SimplexAutomorphism(ABC):
    """
    Base class for all simplex automorphisms

    :param num_classes: The dimension of the simplex vector. This equals 1 + (dimension of the simplex as manifold)
    """

    def __init__(self, num_classes: int):
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
            raise ValueError(
                f"Input has to be from a {self.num_classes - 1} dimensional simplex"
            )
        x = self._transform(x)
        if check_io and not in_simplex(x, self.num_classes):
            raise ValueError(
                f"Bad implementation: Output has to be from a {self.num_classes - 1} dimensional simplex"
            )
        return x.squeeze()


class IdentitySimplexAutomorphism(SimplexAutomorphism):
    def _transform(self, x: np.ndarray) -> np.ndarray:
        return x


class SingleComponentSimplexAutomorphism(SimplexAutomorphism):
    """
    A simplex automorphism resulting from the application of a map on the unit interval to a
    single component of x and normalizing the result.

    :param num_classes:
    :param component: integer in range [0, num_classes - 1], corresponding to the component on which to apply the mapping
    :param mapping: map from the unit interval [0,1] to itself, should be applicable to arrays
    """

    def __init__(
        self,
        num_classes: int,
        component: int,
        mapping: Callable[[np.ndarray], np.ndarray],
    ):
        assert (
            0 <= component < num_classes
        ), "Selected component should be in the range [0, num_classes - 1]"
        self.component = component
        self.mapping = mapping
        super().__init__(num_classes)

    def _transform(self, x: np.ndarray) -> np.ndarray:
        x = x.copy()
        x[:, self.component] = self.mapping(x[:, self.component])
        return x / x.sum(axis=1)[:, None]


class MaxComponentSimplexAutomorphism(SimplexAutomorphism):
    """
    A simplex automorphism resulting from the application of a map on the unit interval to a
    the argmax of x and normalizing the result.

    :param num_classes:
    :param mapping: map from the unit interval [0,1] to itself
    """

    def __init__(self, num_classes, mapping: Callable[[np.ndarray], np.ndarray]):
        self.mapping = mapping
        super().__init__(num_classes)

    def _transform(self, x: np.ndarray) -> np.ndarray:
        x = x.copy()
        argmax = x.argmax(axis=1)[:, None]
        new_values = self.mapping(x.max(axis=1))[:, None]
        np.put_along_axis(x, argmax, new_values, axis=1)
        return x / x.sum(axis=1)[:, None]


class PowerLawSimplexAutomorphism(SimplexAutomorphism):
    """
    An automorphism resulting from taking elementwise powers of the inputs with fixed exponents
    and normalizing the result.

    |
    | *Intuition*:

    If exponents[j] < exponents[i], then the output will be more shifted towards the j-th direction
    than the i-th. If all exponents are equal to some number s, then s>1 means a shift towards the boundary
    of the simplex whereas 0<s<1 means a shift towards the center and s < 0 results in an "antipodal shift".

    :param exponents: numpy array of shape (num_classes, )
    """

    def __init__(self, exponents: np.ndarray):
        self.exponents = exponents
        super().__init__(len(exponents))

    def _transform(self, x: np.ndarray) -> np.ndarray:
        x = np.float_power(x, self.exponents)
        return x / x.sum(axis=1)[:, None]


class RestrictedPowerSimplexAutomorphism(SimplexAutomorphism):
    def __init__(self, exponents: np.ndarray):
        if not np.all(exponents >= 1):
            raise ValueError("Only exponents >= 1 are permitted")
        self.exponents = exponents[None, :]
        super().__init__(len(exponents) + 1)

    def _transform(self, x: np.ndarray) -> np.ndarray:
        x = x.copy()
        x[:, :-1] = np.float_power(x[:, :-1], self.exponents)
        x[:, -1] = 1 - x[:, :-1].sum(axis=1)
        return x
