import pickle
from starter.starter.basemodel import Data
from starter.starter.ml.data import process_data
from starter.starter.ml.model import inference
import pandas as pd


with open('starter/model/gb_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('starter/model/encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

with open('starter/model/lb.pkl', 'rb') as f:
    lb = pickle.load(f)


def infer(data: Data):
    """Run inference for a given data

    Args:
        data (Data): data for inference

    Returns:
        str: prediction
    """

    data = {key.replace('_', '-'): [value] for key, value in data.__dict__.items()}

    row = pd.DataFrame.from_dict(data)

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
        row, categorical_features=cat_features, label=None,
        training=False, encoder=encoder, lb=lb
    )

    pred = inference(model, processed_row)[0]
    if pred == 0:
        return '<=50K'
    if pred == 1:
        return '>50k'
