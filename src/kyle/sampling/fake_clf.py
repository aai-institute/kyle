from abc import ABC, abstractmethod
from typing import Sequence, Union

import numpy as np
import scipy.stats

from kyle.transformations import IdentitySimplexAut, SimplexAut
from kyle.util import sample_index


class FakeClassifier(ABC):
    def __init__(
        self,
        num_classes: int,
        simplex_automorphism: SimplexAut = None,
        check_io=True,
    ):
        if num_classes < 1:
            raise ValueError(f"{self.__class__.__name__} requires at least two classes")
        self.num_classes = num_classes
        self._rng = np.random.default_rng()

        self._simplex_automorphism: SimplexAut = None
        self.set_simplex_automorphism(simplex_automorphism)
        self.check_io = check_io

    # TODO or not TODO: one could get rid of separate SimplexAut. class in favor of passing a function
    #   pro: the function is less verbose to write, easier for user; contra: naming and state become more convoluted
    def set_simplex_automorphism(self, aut: Union[SimplexAut, None]) -> None:
        """
        :param aut: if None, the identity automorphism will be set
        """
        if aut is None:
            aut = IdentitySimplexAut(self.num_classes)
        if aut.num_classes is not None and aut.num_classes != self.num_classes:
            raise ValueError(f"{aut} has wrong number of classes: {aut.num_classes}")
        self._simplex_automorphism = aut

    @abstractmethod
    def sample_confidences(self, n_samples: int) -> np.ndarray:
        ...

    @property
    def simplex_automorphism(self):
        return self._simplex_automorphism

    def get_sample_arrays(self, n_samples: int):
        """
        Get arrays with ground truth and predicted probabilities

        :param n_samples:
        :return: tuple of arrays of shapes (n_samples,), (n_samples, n_classes)
        """
        confidences = self.sample_confidences(n_samples)
        gt_probabilities = self.simplex_automorphism.transform(
            confidences, check_io=self.check_io
        )
        gt_labels = sample_index(gt_probabilities)
        return gt_labels, confidences

    def __str__(self):
        return f"{self.__class__.__name__}_{self.simplex_automorphism}"


class DirichletFC(FakeClassifier):
    def __init__(
        self,
        num_classes: int,
        alpha: Sequence[float] = None,
        simplex_automorphism: SimplexAut = None,
    ):
        super().__init__(num_classes, simplex_automorphism=simplex_automorphism)

        self._alpha: np.ndarray = None
        self.set_alpha(alpha)

    def set_alpha(self, alpha: Union[np.ndarray, None]):
        """
        :param alpha: if None, the default value of [1, ..., 1] will be set
        """
        if alpha is None:
            alpha = np.ones(self.num_classes)
        else:
            alpha = np.array(alpha)
            if not alpha.shape == (self.num_classes,):
                raise ValueError(f"Wrong shape of alpha: {alpha.shape}")
        self._alpha = alpha

    @property
    def alpha(self):
        return self._alpha

    def sample_confidences(self, n_samples: int) -> np.ndarray:
        return self._rng.dirichlet(self.alpha, size=n_samples)

    def pdf(self, confidences, alpha=None):
        if alpha is None:
            alpha = self.alpha
        return scipy.stats.dirichlet.pdf(confidences.T, alpha)


class MultiDirichletFC(FakeClassifier):
    """
    A fake classifier that first draws from a K categorical distribution and based on the result then draws from
    1 of K Dirichlet Distributions of a restricted form.
    The K'th Dirichlet Distribution has parameters of the form:  sigma * {1, 1, ..., alpha_k, 1, 1, ...}; alpha > 1
    where 'alpha_k' is at the k'th position.
    Effectively a distribution with a maximum of variable position and variable variance in each corner of the simplex

    :param num_classes:
    :param alpha: numpy array of shape (num_classes,). k'th entry corresponds to alpha_k for the k'th dirichlet
    :param sigma: numpy array of shape (num_classes,). k'th entry corresponds to sigma for the k'th dirichlet
    :param distribution_weights:
    :param simplex_automorphism:
    """

    def __init__(
        self,
        num_classes: int,
        alpha: Sequence[float] = None,
        sigma: Sequence[float] = None,
        distribution_weights: Sequence[float] = None,
        simplex_automorphism: SimplexAut = None,
    ):
        super().__init__(num_classes, simplex_automorphism=simplex_automorphism)

        self._alpha: np.ndarray = None
        self._sigma: np.ndarray = None
        self._distribution_weights: np.ndarray = None

        self.set_alpha(alpha)
        self.set_sigma(sigma)
        self.set_distribution_weights(distribution_weights)

    @property
    def alpha(self):
        return self._alpha

    def set_alpha(self, alpha: Union[np.ndarray, None]):
        """
        :param alpha: if None, the default value of [1, ..., 1] will be set.
        """
        if alpha is None:
            alpha = np.ones(self.num_classes)
        else:
            alpha = np.array(alpha)
            if not alpha.shape == (self.num_classes,):
                raise ValueError(f"Wrong shape of alpha: {alpha.shape}")
        self._alpha = alpha

    @property
    def sigma(self):
        return self._sigma

    def set_sigma(self, sigma: Union[np.ndarray, None]):
        """
        :param sigma: if None, the default value of [1, ..., 1] will be set
        """
        if sigma is None:
            sigma = np.ones(self.num_classes)
        else:
            sigma = np.array(sigma)
            if not sigma.shape == (self.num_classes,):
                raise ValueError(f"Wrong shape of sigma: {sigma.shape}")
        self._sigma = sigma

    @property
    def distribution_weights(self):
        return self._distribution_weights

    def set_distribution_weights(self, distribution_weights: Union[np.ndarray, None]):
        """
        :param distribution_weights: if None, the default value of [1/num_classes, ..., 1/num_classes] will be set
        """
        if distribution_weights is None:
            distribution_weights = np.ones(self.num_classes) / self.num_classes
        else:
            distribution_weights = np.array(distribution_weights)
            if not distribution_weights.shape == (self.num_classes,):
                raise ValueError(
                    f"Wrong shape of predicted_class_weights: {distribution_weights.shape}"
                )
        self._distribution_weights = distribution_weights / np.sum(distribution_weights)

    def get_parameters(self):
        return self._alpha, self._sigma, self._distribution_weights

    def set_parameters(self, alpha, sigma, distribution_weights):
        self.set_alpha(alpha)
        self.set_sigma(sigma)
        self.set_distribution_weights(distribution_weights)

    def sample_confidences(self, n_samples: int) -> np.ndarray:

        weight_array = np.repeat(self.distribution_weights[None, :], n_samples, axis=0)
        chosen_distributions = sample_index(weight_array)

        confidences = np.zeros((n_samples, self.num_classes))

        for i, chosen_distribution in enumerate(chosen_distributions):
            alpha_vector = np.ones(self.num_classes)
            alpha_vector[chosen_distribution] = self.alpha[chosen_distribution]
            alpha_vector *= self.sigma[chosen_distribution]

            confidences[i, :] = self._rng.dirichlet(alpha_vector)

        return confidences

    def pdf(self, confidences, alpha=None, sigma=None, distribution_weights=None):
        """
        Computes pdf of MultiDirichletFC. Using K categorical distribution to sample from K dirichlet distributions
        is equivalent to sampling from a pdf that is a weighted sum of the K individual dirichlet pdf's
        :param confidences: numpy array of shape (num_classes,) or (num_samples, num_classes)
        :param distribution_weights: numpy array of shape (num_classes,) uses self.distribution_weights if not provided
        :param sigma: numpy array of shape (num_classes,) uses self.sigma if not provided
        :param alpha: numpy array of shape (num_classes,) uses self.alpha if not provided
        """

        if alpha is None:
            alpha = self.alpha
        if sigma is None:
            sigma = self.sigma
        if distribution_weights is None:
            distribution_weights = self.distribution_weights

        confidences = confidences.T

        distributions = np.zeros(confidences.shape)

        for i, (a, s) in enumerate(zip(alpha, sigma)):
            alpha_vector = np.ones(self.num_classes)
            alpha_vector[i] = a
            alpha_vector *= s

            distributions[i] = scipy.stats.dirichlet.pdf(confidences, alpha_vector)

        return np.sum(distribution_weights[:, None] * distributions, axis=0) / np.sum(
            distribution_weights
        )
