from fastapi.testclient import TestClient
from main import app


def test_root():
    with TestClient(app) as testapp:
        response = testapp.get("/")

    assert response.status_code == 200
    assert response.json() == 'Welcome!'


def test_positive_sample():

    data = {
        'age': '39',
        'workclass': 'State-gov',
        'fnlgt': '77516',
        'education': 'Bachelors',
        'education_num': '13',
        'marital_status': 'Never-married',
        'occupation': 'Adm-clerical',
        'relationship': 'Not-in-family',
        'race': 'White',
        'sex': 'Male',
        'capital_gain': '2174',
        'capital_loss': '0',
        'hours_per_week': '0',
        'native_country': 'United-States'
    }

    with TestClient(app) as testapp:
        response = testapp.post("/", json=data)
    
    assert response.status_code == 200
    assert response.json() == '<=50K'


def test_negative_sample():

    data = {
        'age': '40',
        'workclass': 'Private',
        'fnlgt': '193524',
        'education': 'Doctorate',
        'education_num': '16',
        'marital_status': 'Married-civ-spouse',
        'occupation': 'Prof-specialty',
        'relationship': 'Husband',
        'race': 'White',
        'sex': 'Male',
        'capital_gain': '0',
        'capital_loss': '0',
        'hours_per_week': '60',
        'native_country': 'United-States'
    }

    with TestClient(app) as testapp:
        response = testapp.post("/", json=data)

    assert response.status_code == 200
    assert response.json() == '>50k'
