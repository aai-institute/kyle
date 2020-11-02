import pytest
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from kyle.calibration import ModelCalibrator
from kyle.metrics import ECE
from kyle.models import CalibratableModel


@pytest.fixture(scope="module")
def dataset():
    X, y = datasets.make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=7,
        n_redundant=10,
        n_classes=2,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="module")
def uncalibrated_model():
    return MLPClassifier(hidden_layer_sizes=(50, 50, 50))


@pytest.fixture(scope="module")
def calibratable_model(uncalibrated_model):
    return CalibratableModel(uncalibrated_model)


@pytest.fixture(scope="module")
def calibrator(dataset):
    X_train, X_val, y_train, y_val = dataset
    calibrator = ModelCalibrator(X_val, y_val, X_fit=X_train, y_fit=y_train)
    return calibrator


def test_calibrator_integrationTest(calibrator, calibratable_model):
    calibrator.calibrate(calibratable_model, fit=True)
    metric = ECE()
    predicted_probas = calibratable_model.model.predict_proba(calibrator.X_calibrate)
    calibrated_predicted_probas = calibratable_model.predict_proba(
        calibrator.X_calibrate
    )
    assert metric.compute(
        calibrated_predicted_probas, calibrator.y_calibrate
    ) < metric.compute(predicted_probas, calibrator.y_calibrate)
