import pytest
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
import pandas as pd
from starter.ml.data import process_data
from starter.ml.model import inference


@pytest.fixture()
def data():
    data = pd.read_csv('data/census-clean.csv')
    return data


@pytest.fixture()
def X(data):
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    train, _ = train_test_split(data, test_size=0.20)
    X, y, _, _ = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    return X


@pytest.fixture()
def y(data):
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    train, _ = train_test_split(data, test_size=0.20)
    _, y, _, _ = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    return y


@pytest.fixture()
def model(X, y):
    dummy = DummyClassifier()
    dummy.fit(X, y)
    return dummy


def test_data_shape(X, y):
    assert len(X) == len(y)


def test_inference(model, X):
    pred = inference(model, X)

    assert len(X) == len(pred)


def test_age(data):

    assert data["age"].between(3, 150).shape[0] == data.shape[0]
