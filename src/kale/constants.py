from enum import Enum


class Disease(str, Enum):
    healthy = "healthy"
    cold = "cold"
    flue = "flue"
    coronavirus = "coronavirus"
    lung_cancer = "lung_cancer"


class TreatmentCost(float, Enum):
    healthy = 0
    cold = 1
    flue = 1
    coronavirus = 2
    lung_cancer = 3
