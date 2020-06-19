from typing import Iterable

import pandas as pd

from game.datastruct import Patient


def get_all_present_diseases(patients: Iterable[Patient]):
    """

    :param patients:
    :return: set of all diseases for which a confidence was assigned
    """
    present_diseases = set()
    for patient in patients:
        present_diseases = present_diseases.union(patient.confidences.keys())
    return present_diseases


def confidences_df(patients: Iterable[Patient]):
    """
    Data frame giving an overview of the disease->confidence mapping for all patients
    Mainly for visualization purposes.

    :param patients:
    :return: data frame with patient names as columns and diseases as index
    """
    df = pd.DataFrame({"Disease": sorted(get_all_present_diseases(patients))})
    for patient in patients:
        df[patient.name] = df["Disease"].apply(lambda disease: patient.confidences.get(disease, 0))
    return df.set_index("Disease")


def treatment_effects_df(patients: Iterable[Patient]):
    """
    Data frame giving an overview of the disease->treatment_effect mapping for all patients.
    Mainly for visualization purposes.

    :param patients:
    :return: data frame with patient names as columns and diseases as index
    """
    df = pd.DataFrame({"Disease": sorted(get_all_present_diseases(patients))})
    for patient in patients:
        df[patient.name] = df["Disease"].apply(lambda disease: patient.treatment_effects.get(disease, 0))
    return df.set_index("Disease")
