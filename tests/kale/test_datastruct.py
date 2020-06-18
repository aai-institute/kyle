import pytest

from kale.constants import Disease, TreatmentCost
from kale.datastruct import Patient, PatientCollection


@pytest.fixture
def pat1():
    return Patient(name="John", treatment_effect_dict={Disease.healthy: 0, Disease.cold: 3},
                   confidence_dict={Disease.healthy: 0.3, Disease.cold: 0.7}, disease=Disease.cold)


@pytest.fixture
def pat2():
    return Patient(name="Jane", treatment_effect_dict={Disease.healthy: 0, Disease.lung_cancer: 10},
                   confidence_dict={Disease.healthy: 0.2, Disease.lung_cancer: 0.8}, disease=Disease.healthy)


@pytest.fixture
def pat3():
    return Patient(name="Jackson", treatment_effect_dict={Disease.healthy: 0, Disease.lung_cancer: 10},
                   confidence_dict={Disease.healthy: 0.8, Disease.lung_cancer: 0.2}, disease=Disease.lung_cancer)


@pytest.fixture
def patient_collection1(pat1, pat2):
    return PatientCollection(patients=[pat1, pat2], identifier=0)


@pytest.fixture
def patient_collection2(pat1, pat3):
    return PatientCollection(patients=[pat1, pat3], identifier=0)


def test_Patient(pat1, pat2):
    assert pat1.name == "John"
    assert pat1.disease == "cold"
    assert pat1.true_life_gain(Disease.healthy) == 0.0
    assert pat1.true_life_gain(Disease.lung_cancer) == 0.0
    assert pat1.true_life_gain(Disease.cold) == 3.0
    assert pat1.expected_life_gain(Disease.healthy) == 0.0
    assert pat1.expected_life_gain(Disease.lung_cancer) == 0.0
    assert pat1.expected_life_gain(Disease.cold) == 0.7 * 3
    assert pat1.optimal_expected_life_gain() == 0.7 * 3
    assert pat1.maximal_life_gain() == 3.0
    assert pat2.maximal_life_gain() == 0.0
    assert pat1.uuid != pat2.uuid
    assert hash(pat1) != hash(pat2)


def test_PatientCollection_basics(patient_collection1):
    assert patient_collection1.identifier == 0

    # with unbounded costs we just heal the disease
    treatments_dict, expected_life_gain, cost = patient_collection1.get_optimal_treatment()
    assert sorted(treatments_dict.values()) == ["cold", "lung_cancer"]
    assert expected_life_gain == 0.7*3 + 0.8*10
    assert cost == TreatmentCost.cold + TreatmentCost.lung_cancer

    # checking the treatment-evaluating methods
    assert patient_collection1.get_true_life_gain(treatments_dict) == 3.0
    assert patient_collection1.get_treatment_cost(treatments_dict) == cost
    assert patient_collection1.get_expected_life_gain(treatments_dict) == expected_life_gain
    assert patient_collection1.get_counterfactual_optimal_treatment()[1] \
        == patient_collection1.get_maximal_life_gain() == 3.0


def test_PatientCollection_bounded_cost(patient_collection1):
    # adding a hard cost boundary - here we can only heal cold, so we do it
    treatments_dict, expected_life_gain, cost = patient_collection1.get_optimal_treatment(max_cost=2)
    assert sorted(treatments_dict.values()) == ["cold", "healthy"]
    assert expected_life_gain == 0.7 * 3
    assert cost == TreatmentCost.cold

    # if possible, it is more beneficial to heal lung cancer for these patients
    treatments_dict, expected_life_gain, cost = patient_collection1.get_optimal_treatment(max_cost=3)
    assert sorted(treatments_dict.values()) == ["healthy", "lung_cancer"]
    assert expected_life_gain == 0.8 * 10
    assert cost == TreatmentCost.lung_cancer


def test_PatientCollection_nontrivial_optimization(patient_collection2):
    # cost-wise it would be possible to heal cancer instead of cold, so if we knew the true diseases, we would do it
    # NB: since the counterfactual method involves mutating fields, it is good to test it first
    treatments_dict, life_gain, cost = patient_collection2.get_counterfactual_optimal_treatment(max_cost=3)
    assert sorted(treatments_dict.values()) == ["healthy", "lung_cancer"]
    assert life_gain == 10.0
    assert cost == TreatmentCost.lung_cancer

    # in expectation it is more beneficial to heal the cold since 0.7 * 3 > 0.2 * 10
    treatments_dict, expected_life_gain, cost = patient_collection2.get_optimal_treatment(max_cost=3)
    assert sorted(treatments_dict.values()) == ["cold", "healthy"]
    assert expected_life_gain == 0.7 * 3
    assert cost == TreatmentCost.cold

# TODO: multiple important cases are missing
