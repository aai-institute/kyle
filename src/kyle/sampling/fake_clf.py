from abc import ABC, abstractmethod
from typing import Sequence, List, Tuple, Union

import numpy as np

from kyle.transformations import (
    IdentitySimplexAutomorphism,
    SimplexAutomorphism,
    ShiftingSimplexAutomorphism,
)
from kyle.util import in_simplex


class FakeClassifier(ABC):
    def __init__(self, num_classes: int):
        if num_classes < 1:
            raise ValueError(f"{self.__class__.__name__} requires at least two classes")
        self.num_classes = num_classes

    @abstractmethod
    def get_sample(self) -> Tuple[int, np.ndarray]:
        pass

    def get_sample_arrays(self, n_samples: int):
        """
        Get arrays with ground truth and predicted probabilities

        :param n_samples:
        :return: tuple of arrays of shapes (n_samples,), (n_samples, n_classes)
        """
        ground_truth_labels = []
        confidences = []
        for _ in range(n_samples):
            gt_label, confidence_vector = self.get_sample()
            ground_truth_labels.append(gt_label)
            confidences.append(confidence_vector)
        return np.array(ground_truth_labels), np.array(confidences)


class SufficientlyConfidentFC(FakeClassifier):
    """
    A fake classifier for sampling ground truth and class probabilities vectors, see
    `Fake Classifiers <https://gitlab.aai.lab/tl/calibration/texts/-/blob/master/Fake%20Classifiers.tm>`_
    for more details.
    By default instantiated with uniform distributions and trivial simplex automorphisms, these can be adjusted
    after instantiation.


    :param num_classes:
    """

    def __init__(self, num_classes: int):
        super().__init__(num_classes)
        self.predicted_class_weights: np.ndarray = (
            np.ones(self.num_classes) / self.num_classes
        )
        self.alpha = np.ones((self.num_classes, self.num_classes))
        self.simplex_automorphisms: List[SimplexAutomorphism] = [
            IdentitySimplexAutomorphism(self.num_classes)
        ] * self.num_classes
        # pyodide uses an older version of numpy which does not have the default_rng
        # self._rng = np.random.default_rng()
        self._rng = np.random

    def _unit_vector(self, i: int):
        e_i = np.zeros(self.num_classes)
        e_i[i] = 1
        return e_i

    def set_alpha(self, component: int, distribution_alpha: np.ndarray):
        """
        Set alpha parameters for a singe dirichlet distribution

        :param component: thi distribution to modify
        :param distribution_alpha: array of shape (n_classes,) corresponding to the new distribution weights
        :return:
        """
        if not distribution_alpha.shape == (self.num_classes,) or any(
            distribution_alpha < 0
        ):
            raise ValueError("Invalid alpha parameters for dirichlet distribution")
        self.alpha[component] = distribution_alpha

    def set_simplex_automorphism(
        self, component: int, aut: Union[SimplexAutomorphism, None]
    ):
        if aut is None:
            aut = IdentitySimplexAutomorphism(self.num_classes)
        if aut.num_classes != self.num_classes:
            raise ValueError(f"{aut} has wrong number of classes: {aut.num_classes}")
        self.simplex_automorphisms[component] = aut

    def set_predicted_class_weights(self, weights: np.ndarray):
        if not in_simplex(weights, self.num_classes):
            raise ValueError("Invalid weights array: must be in simplex")

    # TODO or not TODO: this could be vectorized, removing the need for get_sample_array
    #   However, this would complicate the code, as the selections of the appropriate distribution_alpha and simplex automorphism
    #   are not vectorized. Since performance is not critical here, maybe vectorization can be postponed
    def get_sample(self):
        predicted_class = self._rng.choice(
            self.num_classes, 1, p=self.predicted_class_weights
        )[0]
        alpha = self.alpha[predicted_class]
        k = self._rng.dirichlet(alpha)
        confidence_vector = ShiftingSimplexAutomorphism(
            self._unit_vector(predicted_class)
        ).transform(k)
        gt_label_weights = self.simplex_automorphisms[predicted_class].transform(
            confidence_vector
        )
        gt_label = self._rng.choice(self.num_classes, 1, p=gt_label_weights)[0]
        return gt_label, confidence_vector


class DirichletFC(FakeClassifier):
    def __init__(
        self,
        num_classes: int,
        alpha: np.ndarray = None,
        simplex_automorphism: SimplexAutomorphism = None,
    ):
        super().__init__(num_classes)
        self.alpha = None
        self.simplex_automorphism = None
        self._rng = np.random.default_rng()

        self.set_simplex_automorphism(simplex_automorphism)
        self.set_alpha(alpha)

    def set_alpha(self, alpha: Union[np.ndarray, None]):
        """
        :param alpha: if None, the default value of [1, ..., 1] will be set
        """
        if alpha is None:
            alpha = np.ones(self.num_classes)
        if not alpha.shape == (self.num_classes,):
            raise ValueError(f"Wrong shape of alpha: {alpha.shape}")
        self.alpha = alpha

    def set_simplex_automorphism(self, aut: Union[SimplexAutomorphism, None]) -> None:
        """
        :param aut: if None, the identity automorphism will be set
        """
        if aut is None:
            aut = IdentitySimplexAutomorphism(self.num_classes)
        if aut.num_classes != self.num_classes:
            raise ValueError(f"{aut} has wrong number of classes: {aut.num_classes}")
        self.simplex_automorphism = aut

    def get_sample(self):
        confidence_vector = self._rng.dirichlet(self.alpha)
        gt_label_weights = self.simplex_automorphism.transform(confidence_vector)
        gt_label = self._rng.choice(self.num_classes, 1, p=gt_label_weights)[0]
        return gt_label, confidence_vector


class SufficientlyConfidentFCBuilder:
    """
    Helper class for instantiating sufficiently confident fake classifiers.

    By default instantiated with uniform distributions and trivial simplex automorphisms, these can be adjusted
    after instantiation.

    :param num_classes: Number of ground truth classes, must be larger than 1
    """

    def __init__(self, num_classes: int):
        self._fc = SufficientlyConfidentFC(num_classes)
        assert (
            num_classes > 1
        ), f"{self.__class__.__name__} requires at least two classes"
        self.num_classes = num_classes
        self.predicted_class_weights: np.ndarray = (
            np.ones(self.num_classes) / self.num_classes
        )
        self.alpha = np.ones((self.num_classes, self.num_classes))
        self.simplex_automorphisms: List[SimplexAutomorphism] = [
            IdentitySimplexAutomorphism(self.num_classes)
        ] * self.num_classes

    def with_predicted_class_weights(self, weights: np.ndarray):
        if not in_simplex(weights, self.num_classes):
            raise ValueError(
                f"Input has to be from a {self.num_classes - 1} dimensional simplex"
            )
        self.predicted_class_weights = weights
        return self

    def with_simplex_automorphisms(
        self, simplex_automorphisms: Sequence[Union[SimplexAutomorphism, None]]
    ):
        assert (
            len(simplex_automorphisms) == self.num_classes
        ), f"Expected {self.num_classes} simplex automorphisms"
        for i, aut in enumerate(simplex_automorphisms):
            self._fc.set_simplex_automorphism(i, aut)
        return self

    def with_alpha(self, alpha: np.ndarray):
        """
        Parameters of the predicted class dirichlet distributions. distribution_alpha[i,j] corresponds to
        the j_th weight of the i_th distribution.

        :param alpha: array of shape (n_classes, n_classes) with semi-positive entries
        :return: patients
        """
        assert alpha.shape == (
            self.num_classes,
            self.num_classes,
        ), f"Wrong input shape: {alpha.shape}"
        self._fc.alpha = alpha
        return self

    def reset(self):
        self._fc = SufficientlyConfidentFC(self.num_classes)
        return self

    def build(self):
        fc = self._fc
        self.reset()
        return fc
