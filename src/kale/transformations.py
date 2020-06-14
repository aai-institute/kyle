from abc import abstractmethod, ABC
from typing import Callable

import numpy as np

from kale.utils import in_simplex


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
        pass

    def transform(self, x: np.ndarray):
        if not in_simplex(self.num_classes, x):
            raise ValueError(f"Input has to be from a {self.num_classes - 1} dimensional simplex")
        x = self._transform(x.copy())
        if not in_simplex(self.num_classes, x):
            raise Exception(f"Bad implementation: Output has to be from a {self.num_classes - 1} dimensional simplex")
        return x


class IdentitySimplexAutomorphism(SimplexAutomorphism):
    def _transform(self, x: np.ndarray) -> np.ndarray:
        return x


class SingleComponentSimplexAutomorphism(SimplexAutomorphism):
    """
    A simplex automorphism resulting from the application of a map on the unit interval to a
    single component of x and normalizing the result.

    :param num_classes:
    :param component: integer in range [0, num_classes - 1], corresponding to the component on which to apply the mapping
    :param mapping: map from the unit interval [0,1] to itself
    """
    def __init__(self, num_classes: int, component: int, mapping: Callable[[float], float]):
        assert 0 <= component < num_classes, "Selected component should be in the range [0, num_classes - 1]"
        self.component = component
        self.mapping = mapping
        super().__init__(num_classes)

    def _transform(self, x: np.ndarray) -> np.ndarray:
        x[self.component] = self.mapping(x[self.component])
        return x/x.sum()


class ScalingSimplexAutomorphism(SimplexAutomorphism):
    """
    An automorphism that scales each axis/class with the corresponding parameter and normalizes the result such
    tha it sums 1. If all scaling parameters are equal, this corresponds to the identity operation.

    :param scaling_parameters: array with positive numbers, one per class
    """

    def __init__(self, scaling_parameters: np.ndarray):
        self.scaling_parameters = scaling_parameters
        super().__init__(len(scaling_parameters))

    def _transform(self, x: np.ndarray) -> np.ndarray:
        x = np.multiply(x, self.scaling_parameters)
        return x / x.sum()


class MaxComponentSimplexAutomorphism(SimplexAutomorphism):
    """
    A simplex automorphism resulting from the application of a map on the unit interval to a
    the argmax of x and normalizing the result.

    :param num_classes:
    :param mapping: map from the unit interval [0,1] to itself
    """
    def __init__(self, num_classes, mapping: Callable[[float], float]):
        self.mapping = mapping
        super().__init__(num_classes)

    def _transform(self, x: np.ndarray) -> np.ndarray:
        i = x.argmax()
        x[i] = self.mapping(x[i])
        return x/x.sum()


class ShiftingSimplexAutomorphism(SimplexAutomorphism):
    def __init__(self, shifting_vector: np.ndarray):
        self.shifting_vector = shifting_vector
        super().__init__(len(shifting_vector))

    def _transform(self, x: np.ndarray) -> np.ndarray:
        x = x + self.shifting_vector
        return x / x.sum()
