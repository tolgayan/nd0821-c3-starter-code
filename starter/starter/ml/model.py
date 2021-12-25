from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV


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
