import logging
import math
from abc import ABC, abstractmethod
from typing import Iterable, Dict, List, Optional

import names
import numpy as np
from pydantic import BaseModel

from game.constants import Disease, TYPICAL_TREATMENT_EFFECTS
from game.datastruct import Patient, PatientCollection
from kale.sampling.fake_clf import FakeClassifier

log = logging.getLogger(__name__)


class Round(PatientCollection):
    max_cost: float
    assigned_treatments: Optional[Dict[Patient, str]] = None
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
        return self.assigned_treatments is not None

    def reset(self):
        self.assigned_treatments = None
        self.results = None

    def play(self, treatments: Dict[Patient, str]):
        """
        Play the round by assigning treatments. Executing this will set the treatments, compute and set results and
        mark the round as played

        :param treatments: dict assigning treated diseases to all patients in the round
            (its keys may form a superset of the patients)
        :return: None
        """
        if self.was_played():
            raise ValueError(f"Round {self.identifier} was already played. "
                            f"You can reassign treatments after resetting it")
        missing_patients = set(self.patients).difference(treatments)
        if missing_patients:
            raise KeyError(f"Invalid treatments in round {self.identifier}: "
                           f"missing assignment for patients: {[pat.name for pat in missing_patients]}")
        unknown_diseases = set(treatments.values()).difference(Disease)
        # this will in fact lead to a key error when computing costs but we might want to have a default cost later
        if unknown_diseases:
            log.warning(f"Treatment assignments for unknown diseases: {unknown_diseases}.")

        treatment_assignments = {}
        for patient in self:
            treatment_assignments[patient] = treatments[patient]
        cost = self.get_treatment_cost(treatment_assignments)
        if cost > self.max_cost:
            raise ValueError(f"Invalid treatments in round {self.identifier}: "
                             f"assigned treatments' cost is {cost}: larger than max_cost {self.max_cost}")

        self.assigned_treatments = treatment_assignments
        cost = self.get_treatment_cost(self.assigned_treatments)
        expected_life_gain = self.get_expected_life_gain(self.assigned_treatments)
        true_life_gain = self.get_true_life_gain(self.assigned_treatments)
        self.results = self.Results(cost=cost, expected_life_gain=expected_life_gain, true_life_gain=true_life_gain)


class PatientProvider(ABC):
    @abstractmethod
    def provide(self, n: int) -> Iterable[Patient]:
        pass


class FakeClassifierPatientProvider(PatientProvider):
    def __init__(self, fake_classifier: FakeClassifier, disease_enum=Disease):
        self.name_provider = names
        self.fake_classifier = fake_classifier
        self.disease_list = list(disease_enum)
        if fake_classifier.num_classes != len(self.disease_list):
            raise ValueError(f"Fake classifier num_classes does not match number of provided diseases")

    def provide(self, n: int) -> Iterable[Patient]:
        for j in range(n):
            name = self.name_provider.get_full_name()
            disease_label, confidence_array = self.fake_classifier.get_sample()
            disease = self.disease_list[disease_label]
            confidences = {}
            treatment_effects = {}
            for i, disease in enumerate(self.disease_list):
                confidences[disease] = confidence_array[i]
                # bringing some per-patient variance to the degree to which medicine is useful
                treatment_effects[disease] = int(TYPICAL_TREATMENT_EFFECTS[disease] * np.random.random() * 2)
            yield Patient(name, treatment_effects, confidences, disease)


# TODO: write tests (once we agree on the interface)
class Game:
    """
    Instances of this class represent a calibration game. The intended gameplay is
        1) start a new round with some number of patients (they will be drawn from the provider)
        2) submit your solution in form of a dict Patient->treated_disease. You must submit an assignment
           for all patients, even if treated_disease is "healthy"
        3) repeat

    The game can be ended manually by calling .end() or automatically. It can be reset using the corresponding
    method.

    :param patient_provider:
    """
    def __init__(self, patient_provider: PatientProvider):
        self.patient_provider = patient_provider
        self.played_rounds = []
        self._current_round: Optional[Round] = None
        self._has_ended = False

    @property
    def current_round(self):
        return self._current_round

    @property
    def has_ended(self):
        return self._has_ended

    def start_new_round(self, n_patients):
        """
        Starts and returns a new round

        :param n_patients:
        :return:
        """
        if self.current_round is not None:
            raise ValueError("An unfinished round already exists, you can access it through .current_round")
        if self._has_ended:
            raise ValueError("Game has already ended. Reset it if you want to start again.")
        round_id = len(self.played_rounds)
        patients = list(self.patient_provider.provide(n_patients))
        self._current_round = Round(patients=patients, identifier=round_id)

    def play_current_round(self, treatments: Dict[Patient, str]):
        if self.current_round is None:
            raise ValueError("No current round exists, you can start a new one with start_new_round.")
        if self._has_ended:
            raise ValueError("Game has already ended. Reset it if you want to start again.")
        self.current_round.play(treatments)
        self.played_rounds.append(self.current_round)
        self._current_round = None

    def restart_current_round(self, n_patients=None):
        """
        Restarts current round and returns it

        :param n_patients: if None, will use the same number of patients as in current round
        :return:
        """
        if self.current_round is None:
            raise ValueError("No current round exists, you can start a new one with start_new_round.")
        if self._has_ended:
            raise ValueError("Game has already ended. Reset it if you want to start again.")

        if n_patients is None:
            n_patients = len(self.current_round)
        patients = list(self.patient_provider.provide(n_patients))
        self._current_round = Round(patients=patients, identifier=self._current_round.identifier)
        return self.current_round.copy()

    def reset(self):
        self._current_round = None
        self._has_ended = False
        self.played_rounds = []

    def end(self):
        self._current_round = None
        self._has_ended = True

    def get_round(self, index: int):
        return self.played_rounds[index]

    # TODO: implement methods for summary and evaluation of the game. We can discuss which ones we need.
    #   They will always boil down to a simple loop over the played rounds, as the rounds already have inbuilt
    #   evaluation methods



