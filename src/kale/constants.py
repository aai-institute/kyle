from enum import Enum


class Disease(str, Enum):
    healthy = "healthy"
    cold = "cold"
    flu = "flu"
    bronchitis = "coronavirus"
    lung_cancer = "lung_cancer"


class TreatmentCost(float, Enum):
    healthy = 0
    cold = 1
    flu = 1
    bronchitis = 2
    lung_cancer = 3
