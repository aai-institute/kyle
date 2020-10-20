from typing import Union
import numpy as np

from kale.metrics.base_calibration_error import BaseCalibrationError
from kale.metrics.dataclasses.kernels import KernelType
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel, linear_kernel


class SKCE(BaseCalibrationError):
    """
    Implementation of Squared Kernel Calibration Error (SKCE) as proposed in [1]_.
    The method is used to measure strong calibration of a classifier via computing a kernel.
    The kernel :math:`k(x_i, x_j)` is matrix valued, and can be obtained by multiplying any scalar kernel (e.g. RBF)
    with :math:`I_m`, where :math:`m` is the number of classes. Formally, the SKCE is
    defined as:

    .. math::

       SKCE = \\sum_{i=1, j=1}^N ({e_Yi - g(X_i)})^T . k(g(X_i), g(X_j)) . (e_Yj - g(X_j)) ,

    where :math:`e_Yi` is the :math:`i^th` unit vector of :math:`m` dimensions and :math:`g(X_i)` is the output
    probability vector of the classifier for sample :math:`i`.

    Parameters
    ----------
    kernel_type : str, default: 'rbf'
        Type of kernel to use.
        Options include the radial-basis kernel (default), laplacian kernel and linear kernel.
        ['rbf', 'laplacian', 'linear'].

    References
    ----------
    .. [1] Widmann, David, Fredrik Lindsten, and Dave Zachariah:
       "Calibration tests in multi-class classification: A unifying framework."
       Advances in Neural Information Processing Systems, 2019.
       `Get source online <https://arxiv.org/pdf/1910.11385.pdf>`_
    """

    def __init__(self, kernel_type: str = KernelType.rbf):
        super(SKCE, self).__init__()
        self.__kernel_type = kernel_type

    @property
    def kernel_type(self):
        return self.__kernel_type

    @kernel_type.setter
    def kernel_type(self, value: str) -> None:
        if value not in KernelType().get_available_kernels():
            raise ValueError("Invalid kernel type. Use KernelType data class for kernel selection.")
        self.__kernel_type = value

    def compute_kernel(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute kernel based on selected type."""
        if self.__kernel_type == 'rbf':
            result = rbf_kernel(x, y)
        elif self.__kernel_type == 'laplacian':
            result = laplacian_kernel(x, y)
        else:
            result = linear_kernel(x, y)

        return result

    def _compute(self, confidences: np.ndarray, ground_truth: np.ndarray, **kwargs) \
            -> Union[float, np.ndarray, ValueError]:
        """
        Measure (mis)calibration by comparing confidence scores and ground truth.
        Labels are assumed to have integer values.

        Parameters
        ----------
        X : np.ndarray, shape=(n_samples, [n_classes])
            NumPy array with confidence values for each prediction.
            1-D for binary classification, 2-D for multi class (softmax).
        y : np.ndarray, shape=(n_samples,)
            NumPy 1-D array with ground truth labels.

        Returns
        -------
        float
            Squared Kernel Calibration Error (SKCE) or ValueError if parameters are invalid.
        """
        identity_matrix = np.identity(confidences.shape[1])
        unit_vectors = np.zeros_like(confidences)
        unit_vectors[np.arange(ground_truth.size), ground_truth] = 1
        error = 0
        kernel_matrix = self.compute_kernel(confidences, confidences)

        # (Squared) Equation (4) from Widmann, Lindsten and Zachariah (NeurIPS 2019)
        diff_vector = unit_vectors - confidences
        diff_vector_transposed = np.transpose(diff_vector)
        for i in range(kernel_matrix.shape[0]):
            diff_i = diff_vector_transposed[:, i]
            for j in range(kernel_matrix.shape[1]):
                error += np.matmul(diff_i,
                                   np.matmul((kernel_matrix[i, j] * identity_matrix),
                                             diff_vector[j]))

        return error / (len(ground_truth) ** 2)
