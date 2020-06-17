import math
from abc import ABC, abstractmethod
from typing import Iterable, Dict, List, Optional

from pydantic import BaseModel

from kale.datastruct import Patient, PatientCollection


class PatientProvider(ABC):
    @abstractmethod
    def provide(self, n: int) -> Iterable[Patient]:
        pass


class Round(PatientCollection):
    max_cost: float
    assigned_treatment_dict: Optional[Dict[Patient, str]] = None
    results: Optional['Round.Results'] = None

    def __init__(self, patients: List[Patient], identifier: int, max_cost=None):
        """

        :param patients:
        :param identifier:
        :param max_cost:
        """
        if max_cost is None:
            max_cost = math.inf
        self.update_forward_refs()  # needed to resolve the internal class Round.Results
        super().__init__(patients, identifier, max_cost=max_cost)

    class Results(BaseModel):
        cost: float
        expected_life_gain: float
        true_life_gain: float

        # just for IDE
        def __init__(self, cost: float, expected_life_gain: float, true_life_gain: float):
            super().__init__(cost=cost, expected_life_gain=expected_life_gain, true_life_gain=true_life_gain)

        class Config:
            allow_mutation = False

    def was_played(self):
        return self.assigned_treatment_dict is not None

    def reset(self):
        self.assigned_treatment_dict = None
        self.results = None

    def play(self, treatment_dict: Dict[Patient, str]):
        """
        Play the round by assigning treatments. Executing this will set the treatments, compute and set results and
        mark the round as played

        :param treatment_dict: dict assigning treated diseases to all patients in the round
            (its keys may form a superset of the patients)
        :return: None
        """
        if self.was_played():
            raise ValueError(f"Round {self.identifier} was already played. "
                            f"You can reassign treatments after resetting it")
        missing_patients = set(self.patients).difference(treatment_dict)
        if missing_patients:
            raise KeyError(f"Invalid treatment_dict in round {self.identifier}: "
                           f"missing assignment for patients: {[pat.name for pat in missing_patients]}")

        treatment_assignments = {}
        for patient in self:
            treatment_assignments[patient] = treatment_dict[patient]
        cost = self.get_treatment_cost(treatment_assignments)
        if cost > self.max_cost:
            raise ValueError(f"Invalid treatment_dict in round {self.identifier}: "
                             f"assigned treatments' cost is {cost}: larger than max_cost {self.max_cost}")

        self.assigned_treatment_dict = treatment_assignments
        cost = self.get_treatment_cost(self.assigned_treatment_dict)
        expected_life_gain = self.get_expected_life_gain(self.assigned_treatment_dict)
        true_life_gain = self.get_true_life_gain(self.assigned_treatment_dict)
        self.results = self.Results(cost=cost, expected_life_gain=expected_life_gain, true_life_gain=true_life_gain)


class Game:
    def __init__(self, patient_provider: PatientProvider):
        self.patient_provider = patient_provider
