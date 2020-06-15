import math
from typing import List, Dict, Union

import numpy as np

from kale.datastruct import NamedStruct, T, NamedStructCollection
from kale.util import iter_param_combinations


class Patient(NamedStruct[Union[int, np.ndarray]]):
    def __init__(self, name: str, disease_gt: int, disease_confidences: np.ndarray, delta_t: np.ndarray):
        super().__init__(name)
        self.delta_label_to_delta = delta_t
        self.disease_label_to_confidence = disease_confidences
        self.disease_gt = disease_gt

    @classmethod
    def _instantiateFromDict(cls, name: str, nameValuesDict: Dict[str, Union[str, Dict]]) -> T:
        return cls(name, **nameValuesDict)

    def _toDict(self) -> Dict[str, Union[int, np.ndarray]]:
        return {"delta_t": self.delta_t, "disease_confidences": self.disease_confidences, "disease_gt": self.disease_gt}

    disease_gt: int

    name: str

    def __post_init__(self):
        if not 0 <= self.disease_gt < len(self.delta_t):
            raise ValueError(f"disease label out of bounds: {self.disease_gt}")
        if not self.disease_confidences.shape == self.delta_t.shape == (len(self.disease_confidences), ):
            raise ValueError(f"wrong input shape: confidences {self.disease_confidences.shape}, "
                             f"delta_t: {self.delta_t.shape}")


class PatientCollection(NamedStructCollection[Patient]):
    @classmethod
    def _instantiateFromDict(cls: T, name: str, nameValuesDict: Dict[str, Union[str, Dict]]) -> T:
        patients = [Patient.fromJSON(v) for v in nameValuesDict.values()]
        return cls(name, *patients)


class Round:
    def __init__(self, patients: List[Patient]):
        self.patients = patients


def get_optimal_disease_distribution(patient_collection: PatientCollection, cost_dict: Dict[int, float], max_cost=None):
    if max_cost is None:
        max_cost = math.inf
    treatments_options_dict = {patient.name: np.arange(len(patient.delta_t)) for patient in patient_collection.values()}

    optimal_cost = math.inf
    optimal_treatments = {patient: cost_dict[0] for patient in patient_collection.values()}
    for selected_treatments_dict in iter_param_combinations(treatments_options_dict):
        cost = sum(cost_dict[v] for v in selected_treatments_dict.values())
        if cost > max_cost:
            continue


