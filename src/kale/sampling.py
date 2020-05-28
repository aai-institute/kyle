from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
import pyro
import pyro.distributions as dist
from torch import tensor


class SimplexAutomorphism(ABC):
    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def _in_simplex(self, x: np.ndarray):
        return len(x) == self.num_classes and np.isclose(sum(x), 1) and all(x >= 0) and all(x <= 1)

    @abstractmethod
    def _transform(self, x: np.ndarray) -> np.ndarray:
        pass

    def transform(self, x: np.ndarray):
        if not self._in_simplex(x):
            raise ValueError(f"Input has to be from a {self.num_classes - 1} dimensional simplex")
        result = self._transform(x)
        if not self._in_simplex(result):
            raise Exception(f"Bad implementation: Output has to be from a {self.num_classes - 1} dimensional simplex")
        return result


class IdentitySimplexAutomorphism(SimplexAutomorphism):
    def _transform(self, x: np.ndarray) -> np.ndarray:
        return x


class ScalingSimplexAutomorphism(SimplexAutomorphism):
    def __init__(self, num_classes: int, scaling_parameters: Sequence[float]):
        assert (l_def := len(scaling_parameters)) == num_classes, \
            f"scaling parameters has wrong number of classes {l_def}"
        self.scaling_parameters = scaling_parameters
        super().__init__(num_classes)

    def _transform(self, x: np.ndarray) -> np.ndarray:
        x = np.array([x[i] * scaling_par for i, scaling_par in enumerate(self.scaling_parameters)])
        return x/x.sum()


class FakeClassifier:
    def __init__(self, num_classes: int):
        assert num_classes > 1, f"{self.__class__.__name__} requires at least two classes"
        self.num_classes = num_classes
        self.predicted_class_categorical = dist.Categorical(tensor([1.0] * self.num_classes))
        self.dirichlet_dists = [dist.Dirichlet(tensor([1 / self.num_classes] * self.num_classes))] * self.num_classes
        self.simplex_automorphisms = [IdentitySimplexAutomorphism(self.num_classes)] * self.num_classes

    def _unit_vector(self, i: int):
        e_i = np.zeros(self.num_classes)
        e_i[i] = 1
        return e_i

    def with_predicted_class_categorical(self, weights: Sequence[float]):
        assert len(weights) == self.num_classes, \
            f"Expected {self.num_classes} probabilities of categorical distribution"
        self.predicted_class_categorical = dist.Categorical(tensor(weights))
        return self

    def with_simplex_automorphisms(self, simplex_automorphisms: Sequence[SimplexAutomorphism]):
        assert len(simplex_automorphisms) == self.num_classes, f"Expected {self.num_classes} simplex automorphisms"
        for i, aut in enumerate(simplex_automorphisms):
            if aut.num_classes != self.num_classes:
                raise ValueError(f"simplex automorphism {i} has wrong number of classes: {aut.num_classes}")
        self.simplex_automorphisms = simplex_automorphisms
        return self

    def with_dirichlet_distributions(self, dirichlet_dists: Sequence[dist.Dirichlet]):
        assert len(dirichlet_dists) == self.num_classes, f"Expected {self.num_classes} dirichlet_distributions"
        for i, dirichlet_dist in enumerate(dirichlet_dists):
            if not dirichlet_dist.shape()[0] == self.num_classes:
                raise ValueError(f"dirichlet distribution {i} has wrong shape: {dirichlet_dist.shape()}")
        self.dirichlet_dists = dirichlet_dists
        return self

    def get_sample(self):
        predicted_class = pyro.sample("predicted_class", self.predicted_class_categorical).item()
        k = pyro.sample("k", self.dirichlet_dists[predicted_class]).numpy()
        probabilities_vector = 1/2 * (k + self._unit_vector(predicted_class))
        ground_truth_label = self.simplex_automorphisms[predicted_class].transform(k).argmax()
        return ground_truth_label, probabilities_vector

