# Script to train machine learning model.


# Add the necessary imports for the starter code.
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
import pickle


# Add code to load in the data.
data = pd.read_csv('data/census-clean.csv')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)
y_test_pred = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, y_test_pred)


model_path = 'model/gb_model.pkl'
encoder_path = 'model/encoder.pkl'
lb_path = 'model/lb.pkl'
model_info_path = 'model/gb_model.info'

with open(model_path, 'wb') as f:
    pickle.dump(model, f)

with open(encoder_path, 'wb') as f:
    pickle.dump(encoder, f)

with open(lb_path, 'wb') as f:
    pickle.dump(lb, f)

with open(model_info_path, 'w') as f:
    f.write('Precision: %.3f\nRecall:%.3f\nfbeta:%.3f' %
            (precision, recall, fbeta))

print('GB model saved to %s' % model_path)
