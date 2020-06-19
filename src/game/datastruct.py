from typing import Dict, List, Iterator, Union
from uuid import UUID, uuid1

import numpy as np
from pydantic import BaseModel, validator, Field

from game.constants import TreatmentCost, Disease
from kale.util import get_first_duplicate


class Patient(BaseModel):
    """
    Representation of a patient.
    This class is designed such that it can be easily serialized to and read from json. Hence the subclassing
    of BaseModel and the dicts as preferred data structure for confidences and treatments.

    :param name:
    :param disease: ground truth for disease
    :param treatment_effects: mapping disease_name -> expected life gain (if sick with that disease and treated)
    :param confidences: mapping disease_name -> confidence of being sick
    """

    name: str
    disease: str
    # unfortunately pydantic does not support defaultdicts and if you try to use them, it blows
    # up into your face by converting them to dicts: https://github.com/samuelcolvin/pydantic/issues/1536
    # We might want to contribute to pydantic and solve this issue
    treatment_effects: Dict[str, float]
    confidences: Dict[str, float]
    # uuid is public because pydantic does not allow private fields without hacking around, see
    # https://github.com/samuelcolvin/pydantic/issues/655
    uuid: UUID = Field(default_factory=uuid1)

    def __init__(self, name: str, treatment_effects: Dict[str, float],
                 confidences: Dict[str, float], disease: str):
        super().__init__(name=name, disease=disease, treatment_effects=treatment_effects,
                         confidences=confidences)

    def __hash__(self):
        return hash(self.uuid)

    def __eq__(self, other: 'PatientCollection'):
        if other.__class__ == self.__class__:
            return self.json() == other.json()
        return False

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
        self._validate_disease(treated_disease)
        return self.confidences[treated_disease] * self.treatment_effects[treated_disease]

    def optimal_expected_life_gain(self):
        return max([self.expected_life_gain(treated_disease) for treated_disease in self.confidences])

    def true_life_gain(self, treated_disease: str):
        self._validate_disease(treated_disease)
        if self.disease == treated_disease:
            return self.treatment_effects[treated_disease]
        return 0.0

    def maximal_life_gain(self):
        return self.treatment_effects[self.disease]

    def _validate_disease(self, treated_disease: str):
        if treated_disease not in self.confidences:
            raise KeyError(f"Unexpected treated disease for patient {self.name}: "
                           f"no confidence value for {treated_disease}")


class PatientCollection(BaseModel):
    """
    Representation of a patient collection.
    This class is designed such that it can be easily serialized to and read from json. Hence the subclassing
    of BaseModel and the wrapping of a simple list.

    :param patients: List of patients with unique names
    :param identifier:
    """
    patients: List[Patient]
    identifier: Union[UUID, int]
    uuid: UUID = Field(default_factory=uuid1)

    def __init__(self, patients: List[Patient], identifier: Union[UUID, int] = None, **kwargs):
        if identifier is None:
            identifier = uuid1()
        super().__init__(patients=patients, identifier=identifier, **kwargs)

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, item):
        return self.patients[item]

    def __iter__(self) -> Iterator[Patient]:
        return self.patients.__iter__()

    def __hash__(self):
        return hash(self.uuid)

    def __eq__(self, other: 'PatientCollection'):
        if other.__class__ == self.__class__:
            return self.json() == other.json()
        return False

    @validator("patients")
    def _patients_validator(cls, patients: List[Patient], values):
        duplicate_name = get_first_duplicate([patient.name for patient in patients])
        if duplicate_name is not None:
            raise ValueError(f"PatientCollection {values['identifier']}: Duplicate patient name: {duplicate_name}")
        return patients

    def expected_life_gain(self, treatments: Dict[Patient, str]) -> float:
        """
        Expected value for the life gain due to the treated diseases based on the patients' disease confidences

        :param treatments: dict mapping patient to treated_disease (its keys may form a superset of the collection)
        :return:
        """
        return sum([patient.expected_life_gain(treatments[patient]) for patient in self])

    def true_life_gain(self, treatments: Dict[Patient, str]) -> float:
        """
        True value for the life gain due to the treated diseases based on the patients' disease ground truths

        :param treatments: dict mapping patient to treated_disease (its keys may form a superset of the collection)
        :return:
        """
        return sum([patient.true_life_gain(treatments[patient]) for patient in self])

    def treatment_cost(self, treatments: Dict[Patient, str], treatment_cost=TreatmentCost) -> float:
        """
        Total cost of the assigned treatments

        :param treatments: dict mapping patient to treated_disease (its keys may form a superset of the collection)
        :param treatment_cost: enum representing treated_disease costs
        :return:
        """
        return sum([treatment_cost[treatments[patient]] for patient in self])

    def maximal_life_gain(self):
        """
        The maximal possible life gain obtained by treating all patients correctly

        :return:
        """
        return sum([patient.maximal_life_gain() for patient in self])
