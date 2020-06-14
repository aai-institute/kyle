from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
import pyro
import pyro.distributions as dist
import torch
from torch import tensor


class SimplexAutomorphism(ABC):
    """
    Base class for all simplex automorphisms

    :param num_classes: The dimension of the simplex vector. This equals 1 + (dimension of the simplex as manifold)
    """
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
        return x/x.sum()


class ShiftingSimplexAutomorphism(SimplexAutomorphism):
    def __init__(self, shifting_vector: np.ndarray):
        self.shifting_vector = shifting_vector
        self._norm_factor = 1 + self.shifting_vector.sum()
        super().__init__(len(shifting_vector))

    def _transform(self, x: np.ndarray) -> np.ndarray:
        x = x + self.shifting_vector
        return x/self._norm_factor


class FakeClassifier:
    """
    A fake classifier for sampling ground truth and class probabilities vectors,
    see `Fake Classifiers <https://gitlab.aai.lab/tl/calibration/texts/-/blob/master/Fake%20Classifiers.tm>`_ for more details.
    By default instantiated with uniform distributions and trivial simplex automorphisms, these can be adjusted
    after instantiation.

    :param num_classes: Number of ground truth classes, must be larger than 1
    """
    def __init__(self, num_classes: int):
        assert num_classes > 1, f"{self.__class__.__name__} requires at least two classes"
        self.num_classes = num_classes
        self.predicted_class_categorical = dist.Categorical(torch.ones(self.num_classes))
        self.dirichlet_dists = [dist.Dirichlet(torch.ones(self.num_classes))] * self.num_classes
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

    # TODO or not TODO: this could be vectorized, removing the need for get_sample_array
    #   However, this would complicate the code, as self.dirichlet_dists[predicted_class] is not vectorized.
    #   Since performance is not critical here, maybe vectorization would be an overkill
    def get_sample(self):
        predicted_class = pyro.sample("predicted_class", self.predicted_class_categorical).item()
        k = pyro.sample("k", self.dirichlet_dists[predicted_class]).numpy()
        confidence_vector = ShiftingSimplexAutomorphism(self._unit_vector(predicted_class)).transform(k)
        gt_distribution = self.simplex_automorphisms[predicted_class].transform(confidence_vector)
        gt_categorical = dist.Categorical(tensor(gt_distribution))
        gt_label = pyro.sample("gt_categorical", gt_categorical).item()
        return gt_label, confidence_vector

    def get_sample_arrays(self, n_samples: int):
        """
        Get arrays with ground truth and predicted probabilities

        :param n_samples:
        :return: tuple of arrays of shapes (n_samples,), (n_samples, n_classes)
        """
        ground_truth_labels = []
        probabilities_vectors = []
        for _ in range(n_samples):
            gt_label, proba_vector = self.get_sample()
            ground_truth_labels.append(gt_label)
            probabilities_vectors.append(proba_vector)
        return np.array(ground_truth_labels), np.array(probabilities_vectors)


