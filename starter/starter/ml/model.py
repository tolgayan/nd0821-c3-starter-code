from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from ml.data import process_data


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = GradientBoostingClassifier()

    hyperparam_space = {
        "learning_rate": (1e-2, 1e-3, 1e-4),
        "n_estimators": (10, 50, 100),
        "max_depth": [5, 10]
    }

    classifier = GridSearchCV(model, hyperparam_space, n_jobs=-1, verbose=2)
    classifier.fit(X_train, y_train)

    best_model = classifier.best_estimator_
    picked_hyperparams = classifier.best_params_
    best_score = classifier.best_score_

    print('Best model: %s Hyperparams: %s best score: %s' %
          (best_model, picked_hyperparams, best_score))

    return best_model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def test_slice(model, data, feature_col, encoder, lb):
    """Slice data based on the different values of a given feature, and test the model for each slice.

    Args:
        model (???): ML model to evaluate
        data (pd.DataFrame): data to slice
        feature_col (str): name of the column to slice
        encoder (???): Model encoder object
        lb (???): Model lb object
    """

    with open('model/%s_slice_output.txt' % feature_col, 'a') as f:
        f.write('Model metrics for %s features\n\n\n' % feature_col)

    for feature_val in data[feature_col].unique():
        sliced_data = data[data[feature_col] == feature_val]

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

        processed_data, y, _, _ = process_data(
            sliced_data, categorical_features=cat_features, label='salary',
            training=False, encoder=encoder, lb=lb
        )

        preds = inference(model, processed_data)
        precision, recall, fbeta = compute_model_metrics(y, preds)

        with open('model/%s_slice_output.txt' % feature_col, 'a') as f:
            f.write('Feature value: %s\n' % feature_val)
            f.write('Precision: %s\n' % precision)
            f.write('Recall: %s\n' % recall)
            f.write('F1 Beta: %s\n' % fbeta)
            f.write('\n\n' % fbeta)
