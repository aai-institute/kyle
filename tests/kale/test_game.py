import pytest

from kale.constants import Disease, TreatmentCost
from kale.datastruct import Patient
from kale.game import Round, FakeClassifierPatientProvider
from kale.sampling.fake_clf import DirichletFC


@pytest.fixture
def round0():
    pat1 = Patient(name="John", treatment_effect_dict={Disease.healthy: 0, Disease.cold: 3},
                   confidence_dict={Disease.healthy: 0.3, Disease.cold: 0.7}, disease=Disease.cold)
    pat2 = Patient(name="Jane", treatment_effect_dict={Disease.healthy: 0, Disease.lung_cancer: 10},
                   confidence_dict={Disease.healthy: 0.2, Disease.lung_cancer: 0.8}, disease=Disease.healthy)
    return Round(patients=[pat1, pat2], identifier=0, max_cost=1)


@pytest.fixture
def external_patient():
    return Patient(name="Jackson", treatment_effect_dict={Disease.healthy: 0, Disease.lung_cancer: 10},
                   confidence_dict={Disease.healthy: 0.8, Disease.lung_cancer: 0.2}, disease=Disease.lung_cancer)


@pytest.fixture
def patient_provider():
    return FakeClassifierPatientProvider(DirichletFC(len(list(Disease))))

def test_Round(round0, external_patient):
    assert round0.identifier == 0
    assert round0[0].name == "John"
    assert round0.max_cost == 1
    assert len(round0) == 2
    assert not round0.was_played()
    assert round0.results is None

    # playing the round
    valid_treatment_dict = {round0[0]: Disease.cold, round0[1]: Disease.healthy}
    round0.play({round0[0]: Disease.cold, round0[1]: Disease.healthy})
    assert round0.was_played()
    assert round0.results is not None
    assert round0.results.cost == TreatmentCost.cold
    assert round0.results.expected_life_gain == 0.7 * 3
    assert round0.results.true_life_gain == 3.0
    assert round0.assigned_treatment_dict == valid_treatment_dict
    with pytest.raises(ValueError):
        round0.play(valid_treatment_dict)

    # testing reset
    round0.reset()
    assert round0.results is None
    assert round0.assigned_treatment_dict is None
    round0.play(valid_treatment_dict)

    # testing invalid input
    round0.reset()
    with pytest.raises(KeyError):
        round0.play({round0[0]: Disease.cold, external_patient: Disease.healthy})  # we miss patient Jane
    with pytest.raises(ValueError):
        round0.play({round0[0]: Disease.cold, round0[1]: Disease.lung_cancer})  # too expensive


def test_PatientProvider(patient_provider):
    patients = list(patient_provider.provide(2))
    assert len(list(patient_provider.provide(2))) == 2
    Round(patients, identifier=0)  # test that a round can be constructed
    with pytest.raises(ValueError):
        FakeClassifierPatientProvider(DirichletFC(2))  # wrong number of classes
