import copy
import math
from typing import Tuple, Dict, List, Iterable

from game.constants import TreatmentCost, Disease
from game.datastruct import Patient
from kale.util import iter_param_combinations


def counterfactual_optimal_treatment(patients: Iterable[Patient], max_cost=None, treatment_cost=TreatmentCost):
    """
    Finding the assignment patient->treated_disease that maximizes the true life gain for all patients.
    The solution is based on ground truth for patients' diseases, confidences are not taken into account here

    :param patients:
    :param max_cost: maximal cost of all prescribed treatments
    :param treatment_cost: enum representing treated_disease costs
    :return: tuple (treatments, life_gain, total_cost) where treatments is a dict
        mapping patients to treated_disease
    """
    modified_patients = []
    for pat in patients:
        pat = copy.deepcopy(pat)
        # if pat.disease = healthy the second value wins, so everything works as indented
        pat.confidences = {Disease.healthy: 0, pat.disease: 1.0}
        modified_patients.append(pat)
    return optimal_treatment(modified_patients, max_cost=max_cost, treatment_cost=treatment_cost)


# TODO or not TODO: this is a suboptimal brute-force approach
def optimal_treatment(patients: Iterable[Patient], max_cost=None, treatment_cost=TreatmentCost) \
        -> Tuple[Dict[Patient, str], float, float]:
    """
    Finding the assignment patient->treated_disease that maximizes the total *expected* life gain for all patients.
    The ground truth for patients' diseases is not taken into account here

    :param patients:
    :param max_cost: maximal cost of all prescribed treatments
    :param treatment_cost: enum representing treated_disease costs
    :return: tuple (treatments, expected_life_gain, total_cost) where treatments
        is a dict mapping patient to treated_disease
    """
    if max_cost is None:
        max_cost = math.inf

    treated_disease_options: Dict[Patient, List[str]] = {}
    optimal_treatments: Dict[Patient, str] = {}
    for patient in patients:
        treated_disease_options[patient] = list(patient.confidences.keys())
        optimal_treatments[patient] = Disease.healthy.value  # initialize by recommending no treated_disease

    optimal_life_gain = -math.inf
    optimal_treatment_cost = 0.0
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
