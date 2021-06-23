from abc import ABC, abstractmethod
from typing import Sequence, Union

import numpy as np
import scipy.optimize
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

    def fit(self, confidences, initial_alpha=None, alpha_bounds=None, **kwargs):
        """
        Fits the dirichlet fake classifier to the provided confidence distribution using maximum likelihood estimation
        and sets the fake classifier parameters to the best fit parameters

        :param confidences: Numpy array of shape (num_samples, num_classes);
                            confidence distribution to fit classifier to
        :param initial_alpha: Float; Initial guess for fitting alpha parameters
        :param alpha_bounds: Tuple, (lower_bound, upper_bound); Bounds for fitting alpha parameters. A lower/upper bound
                             of None corresponds to unbounded parameter
        :param kwargs: passed to ``scipy.optimize.minimize``
        :return:
        """
        if initial_alpha is None:
            initial_alpha = self.alpha

        if alpha_bounds is None:
            alpha_bounds = (0.0001, None)

        # rescale confidences to avoid divergences on sides of simplex and renormalize
        confidences = (
            confidences * (confidences.shape[0] - 1) + 1 / self.num_classes
        ) / confidences.shape[0]
        confidences = confidences / np.sum(confidences, axis=1)[:, None]

        alpha_bounds = [alpha_bounds] * self.num_classes

        nll = lambda parm: -np.sum(np.log(self.pdf(confidences, parm)))
        mle_fit = scipy.optimize.minimize(
            nll, initial_alpha, bounds=alpha_bounds, **kwargs
        )
        self.set_alpha(mle_fit.x)

        return mle_fit


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
    :param distribution_weights: numpy array of shape (num_classes,). Probabilities used for drawing from K Categorical
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

    def fit(
        self,
        confidences,
        initial_parameters=None,
        parameter_bounds=None,
        simplified_fitting=True,
        **kwargs,
    ):
        """
        Fits a Multi-Dirichlet fake classifier to the provided confidence distribution using maximum likeihood
        estimation and sets the fake classifier parameters to the best fit parameters.
        If simplified_fitting is set to False all parameters of the fake classifier are fit directly via MLE
        If simplified_fitting is set to True each dirichlet is fit separately. Alpha and Sigma of the k'th dirichlet
        are fit to the subset of the confidences that predict the k'th class, i.e. for which argmax(c) = k. The
        distribution weights are not fit, but estimated from the predicted class probabilities of the confidence
        distribution.

        :param confidences: Numpy array of shape (num_samples, num_classes);
                            confidence distribution to fit classifier to
        :param initial_parameters: Numpy array of shape (3,) ((2,) for simplified_fitting=True)
                       Corresponds to initial guesses for each parameter 'class' alpha, sigma and distribution_weights
                       If None, uses [1, 1, 1/num_classes]
        :param parameter_bounds: Sequence of 3 (2 for simplified_fitting=True) tuples (lower_bound, upper_bound)
                        Corresponds to the bounds on each parameter 'class',  alpha, sigma and distribution_weights
                        A lower/upper bound of None corresponds to unbounded parameters
                        If None, uses intervals [(0, + infinity), (0, + infinity), (0,1)]
        :param simplified_fitting: If False directly fits Multi-Dirichlet FC to confidence distribution
                                   If True fits each dirichlet separately. Only fits alpha and sigma, not
                                   distribution_weights
        :param kwargs: passed to ``scipy.optimize.minimize``
        :return: If simplfied_fitting=False: scipy OptimizeResult
                 If simplified_fitting=True: List of num_classes OptimizeResults, one for each separate dirichlet fit
        """

        # rescale confidences to avoid divergences on sides of simplex and renormalize
        confidences = (
            confidences * (confidences.shape[0] - 1) + 1 / self.num_classes
        ) / confidences.shape[0]
        confidences = confidences / np.sum(confidences, axis=1)[:, None]

        if not simplified_fitting:
            if initial_parameters is None:
                initial_parameters = np.array([1, 1, 1 / self.num_classes])
            if parameter_bounds is None:
                # dirichlet distribution undefined for alpha/sigma parameters exactly = 0
                parameter_bounds = [(0.0001, None)] * 2 + [(0, 1)]

            # scipy requires an initial guess and a bound (lower, upper) for each parameter
            # not just each parameter class
            initial_parameters = np.repeat(initial_parameters, self.num_classes)
            parameter_bounds = [
                pair for pair in parameter_bounds for i in range(self.num_classes)
            ]

            nll = lambda parms: -np.sum(
                np.log(self.pdf(confidences, *np.split(parms, 3)))
            )
            mle_fit = scipy.optimize.minimize(
                nll, initial_parameters, bounds=parameter_bounds
            )
            self.set_parameters(*np.split(mle_fit.x, 3))

            return mle_fit

        if simplified_fitting:
            if initial_parameters is None:
                initial_parameters = np.array([1, 1])
            if parameter_bounds is None:
                # dirichlet distribution undefined for alpha/sigma parameters exactly = 0
                parameter_bounds = [(0.0001, None)] * 2

            predicted_class = np.argmax(confidences, axis=1)
            class_split_confidences = [
                confidences[predicted_class == i, :] for i in range(self.num_classes)
            ]

            estimated_distribution_weights = [
                k_class_conf.shape[0] for k_class_conf in class_split_confidences
            ]
            estimated_distribution_weights = estimated_distribution_weights / np.sum(
                estimated_distribution_weights
            )

            mle_fits = []

            for k, k_class_confidences in enumerate(class_split_confidences):

                def k_dir_nll(alpha_k, sigma_k):
                    alpha = np.ones(self.num_classes)
                    alpha[k] = alpha_k
                    sigma = np.ones(self.num_classes)
                    sigma[k] = sigma_k
                    # 'isolate' the k'th dirichlet distribution
                    distribution_weights = np.zeros(self.num_classes)
                    distribution_weights[k] = 1
                    return -np.sum(
                        np.log(
                            self.pdf(
                                k_class_confidences, alpha, sigma, distribution_weights
                            )
                        )
                    )

                k_initial_parameters = initial_parameters
                k_parameter_bounds = parameter_bounds

                k_dir_mle_fit = scipy.optimize.minimize(
                    lambda parms: k_dir_nll(*parms),
                    k_initial_parameters,
                    bounds=k_parameter_bounds,
                    **kwargs,
                )
                mle_fits.append(k_dir_mle_fit)

            self.set_alpha(np.array([k_mle_fit.x[0] for k_mle_fit in mle_fits]))
            self.set_sigma(np.array([k_mle_fit.x[1] for k_mle_fit in mle_fits]))
            self.set_distribution_weights(estimated_distribution_weights)

        return mle_fits
