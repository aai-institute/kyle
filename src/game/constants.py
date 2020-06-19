from enum import Enum


class Disease(str, Enum):
    healthy = "healthy"
    cold = "cold"
    flu = "flu"
    bronchitis = "bronchitis"
    lung_cancer = "lung_cancer"


class TreatmentCost(float, Enum):
    healthy = 0
    cold = 1
    flu = 1
    bronchitis = 2
    lung_cancer = 3


# values in months
TYPICAL_TREATMENT_EFFECT_DICT = {
    "healthy": 0.0,
    "cold": 12,
    "flu": 24,
    "bronchitis": 36,
    "lung_cancer": 120,
}
