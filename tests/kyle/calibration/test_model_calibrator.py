import pytest
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.model_selection import train_test_split

from kyle.calibration import ModelCalibrator
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
def uncalibrated_model(dataset):
    X_train, _, y_train, _ = dataset
    model = SVC(max_iter=10000, probability=True)
    model.fit(X_train, y_train)

    return model


@pytest.fixture(scope="module")
def calibratable_model(uncalibrated_model):
    return CalibratableModel(uncalibrated_model)


@pytest.fixture(scope="module")
def calibrator(dataset, calibratable_model):
    _, X_test, _, y_test = dataset
    calibrator = ModelCalibrator(calibratable_model, X_test, y_test)

    return calibrator


def test_calibrator_CalibrateOnValSetAfterSettingValSet(dataset, calibrator):
    _, X_val, _, y_val = dataset
    calibrator.set_validation_data(X_val, y_val)
    assert calibrator.calibrate(fit=True) is None


def test_calibrator_errorCalibrateOnValSetWithoutSettingValSet(dataset, calibrator):
    calibrator.set_validation_data(None, None)
    with pytest.raises(AttributeError):
        assert calibrator.calibrate(fit=True)
