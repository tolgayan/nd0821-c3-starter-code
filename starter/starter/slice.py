import pickle
from ml.data import process_data
from ml.model import test_slice
import pandas as pd


with open('model/gb_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

with open('model/lb.pkl', 'rb') as f:
    lb = pickle.load(f)


def slice(data, feature_col):
    """Slice data based on the different values of a given feature, and save the metrics
    for each individual value.

    Args:
        data (pd.DataFrame): data to slice
        feature_col (str): column name
    """

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

    processed_row, _, _, _ = process_data(
        data, categorical_features=cat_features, label=None,
        training=False, encoder=encoder, lb=lb
    )

    test_slice(model=model, data=data, feature_col=feature_col, encoder=encoder, lb=lb)


data = pd.read_csv('data/census-clean.csv')

slice(data, feature_col='education')
slice(data, feature_col='education')
slice(data, feature_col='education')
