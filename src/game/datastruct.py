import copy
import math
from typing import Dict, List, Iterator, Union
from uuid import UUID, uuid1

import numpy as np
import pandas as pd
from pydantic import BaseModel, validator, Field

from game.constants import TreatmentCost, Disease
from kale.util import get_first_duplicate, iter_param_combinations


class Patient(BaseModel):
    name: str
    disease: str
    # unfortunately pydantic does not support defaultdicts and if you try to use them, it blows
    # up into your face by converting them to dicts: https://github.com/samuelcolvin/pydantic/issues/1536
    # We might want to contribute to pydantic and solve this issue
    treatment_effects: Dict[str, float]
    confidences: Dict[str, float]
    uuid: UUID = Field(default_factory=uuid1)

    def __init__(self, name: str, treatment_effects: Dict[str, float],
                 confidences: Dict[str, float], disease: str):
        """

        :param name:
        :param disease: ground truth for disease
        :param treatment_effects: mapping disease_name -> expected life gain (if sick with that disease and treated)
        :param confidences: mapping disease_name -> confidence of being sick
        """
        super().__init__(name=name, disease=disease, treatment_effects=treatment_effects,
                         confidences=confidences)

    @validator("confidences")
    def _confidence_validator(cls, v: Dict[str, float], values):
        if "treatment_effects" in values:
            missing_deltas = set(v).difference(values["treatment_effects"])
            if missing_deltas:
                raise ValueError(f"Patient {values['name']}: Missing deltas for diseases: {missing_deltas}")
        if not np.isclose(sum(v.values()), 1.0):
            raise ValueError(f"Patient {values['name']}: Confidences do not sum to 1")
        if values["disease"] not in v:
            raise ValueError(f"Patient {values['name']}: Missing confidence for ground truth disease: {v}")
        if Disease.healthy not in v:
            raise ValueError(f"Patient {values['name']}: Missing confidence for being healthy")
        return v

    def expected_life_gain(self, treated_disease: str):
        if treated_disease not in self.confidences:
            return 0.0
        return self.confidences[treated_disease] * self.treatment_effects[treated_disease]

    def optimal_expected_life_gain(self):
        return max([self.expected_life_gain(treated_disease) for treated_disease in self.confidences])

    def true_life_gain(self, treated_disease: str):
        if self.disease == treated_disease:
            return self.treatment_effects[treated_disease]
        return 0.0

    def maximal_life_gain(self):
        return self.treatment_effects[self.disease]

    def __hash__(self):
        return hash(self.uuid)


class PatientCollection(BaseModel):
    patients: List[Patient]
    identifier: Union[UUID, int]
    present_diseases: List[str] = None

    def __init__(self, patients: List[Patient], identifier: Union[UUID, int] = None, **kwargs):
        if identifier is None:
            identifier = uuid1()
        present_diseases = set()
        for patient in patients:
            present_diseases = present_diseases.union(patient.confidences.keys())
        super().__init__(patients=patients, identifier=identifier, present_diseases=sorted(present_diseases), **kwargs)

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, item):
        return self.patients[item]

    def __iter__(self) -> Iterator[Patient]:
        return self.patients.__iter__()

    @validator("patients")
    def _patients_validator(cls, patients: List[Patient], values):
        duplicate_name = get_first_duplicate([patient.name for patient in patients])
        if duplicate_name is not None:
            raise ValueError(f"PatientCollection {values['identifier']}: Duplicate patient name: {duplicate_name}")
        return patients

    # TODO or not TODO: this is a suboptimal brute-force approach
    def get_optimal_treatment(self, max_cost=None, treatment_cost=TreatmentCost):
        """
        Finding the assignment patient->treated_disease that maximizes the total *expected* life gain for all patients.
        The ground truth for patients' diseases is not taken into account here

        :param max_cost: maximal cost of all prescribed treatments
        :param treatment_cost: enum representing treated_disease costs
        :return: tuple (treatments, expected_life_gain, total_cost) where treatments
            is a dict mapping patients to the disease to be treated for
        """
        if max_cost is None:
            max_cost = math.inf

        treated_disease_options: Dict[Patient, List[str]] = {}
        optimal_treatments: Dict[Patient, str] = {}
        for patient in self:
            treated_disease_options[patient] = list(patient.confidences.keys())
            optimal_treatments[patient] = Disease.healthy.value  # initialize by recommending no treated_disease

        optimal_life_gain: float = -math.inf
        optimal_treatment_cost: float = 0.0
        for treatments in iter_param_combinations(treated_disease_options):
            cost = 0
            expected_life_gain = 0
            for pat, treated_disease in treatments.items():
                cost += treatment_cost[treated_disease].value
                if cost > max_cost:  # no need to continue the inner loop
                    break
                expected_life_gain += pat.expected_life_gain(treated_disease)
            if cost > max_cost or expected_life_gain < optimal_life_gain:
                continue

            optimal_life_gain = expected_life_gain
            optimal_treatments = treatments
            optimal_treatment_cost = cost

        return optimal_treatments, optimal_life_gain, optimal_treatment_cost

    def get_expected_life_gain(self, treatments: Dict[Patient, str]) -> float:
        """
        Expected value for the life gain due to the treated diseases based on the patients' disease confidences

        :param treatments: dict assigning treated diseases to all patients in the collection
            (its keys may form a superset of the collection)
        :return:
        """
        return sum([patient.expected_life_gain(treatments[patient]) for patient in self])

    def get_true_life_gain(self, treatments: Dict[Patient, str]) -> float:
        """
        True value for the life gain due to the treated diseases based on the patients' disease ground truths

        :param treatments: dict assigning treated diseases to all patients in the collection
            (its keys may form a superset of the collection)
        :return:
        """
        return sum([patient.true_life_gain(treatments[patient]) for patient in self])

    def get_treatment_cost(self, treatments: Dict[Patient, str], treatment_cost=TreatmentCost) -> float:
        """
        Total cost of the assigned treatments

        :param treatments: dict assigning treated diseases to all patients in the collection
            (its keys may form a superset of the collection)
        :param treatment_cost: enum representing treated_disease costs
        :return:
        """
        return sum([treatment_cost[treatments[patient]] for patient in self])

    def get_maximal_life_gain(self):
        """
        The maximal possible life gain obtained by treating all patients correctly

        :return:
        """
        return sum([patient.maximal_life_gain() for patient in self])

    def get_counterfactual_optimal_treatment(self, max_cost=None, treatment_cost=TreatmentCost):
        """
        Finding the assignment patient->treated_disease that maximizes the true life gain for all patients.
        The solution is based on ground truth for patients' diseases, confidences are not taken into account here

        :param max_cost: maximal cost of all prescribed treatments
        :param treatment_cost: enum representing treated_disease costs
        :return: tuple (treatments, life_gain, total_cost) where treatments
            maps patients to the disease to be treated for
        """
        counterfactual_collection = copy.deepcopy(self)
        for pat in counterfactual_collection:
            # if pat.disease = healthy the second value wins, so everything works as indented
            pat.confidences = {Disease.healthy: 0, pat.disease: 1.0}
        return counterfactual_collection.get_optimal_treatment(max_cost=max_cost, treatment_cost=treatment_cost)

    def confidences_df(self):
        df = pd.DataFrame({"Disease": self.present_diseases})
        for patient in self:
            df[patient.name] = df["Disease"].apply(lambda disease: patient.confidences.get(disease, 0))
        return df.set_index("Disease")

    def treatment_effects_df(self):
        df = pd.DataFrame({"Disease": self.present_diseases})
        for patient in self:
            df[patient.name] = df["Disease"].apply(lambda disease: patient.treatment_effects.get(disease, 0))
        return df.set_index("Disease")
