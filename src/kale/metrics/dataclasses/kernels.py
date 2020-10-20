from dataclasses import dataclass


@dataclass
class KernelType:
    """
    Data class containing available kernel options for calculating Squared Kernel Calibration Error (SKCE).
    """
    rbf: str = "rbf"
    linear: str = "linear"
    laplacian: str = "laplacian"

    def get_available_kernels(self):
        return [i for i in self.__dict__.keys() if i[:1] != '_']
