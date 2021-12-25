# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Emre Tolga Ayan created this model. It is a gradient boosting classifier model, trained with scikit learn 1.0.1.
The hyperparameters are picked using grid search. 

Hyperparam Search space:
    learning_rate: (1e-2, 1e-3, 1e-4)
    n_estimators: (10, 50, 100)
    max_depth: [5, 10]

Best Hyperparameters:
    learning_rate: 0.01 
    max_depth: 10
    n_estimators': 100
    best score: 0.8537695314731


## Intended Use

The model should be used to predict salary of a person based on his or her personal information.

## Training Data

Data is retrieved from https://archive.ics.uci.edu/ml/datasets/census+income. 
There are 8 categorical and 6 numeric features about people. 

%80 of the data is used for training. 


## Evaluation Data

%20 of the data is used for evaulation. 


## Metrics

Precision: 0.861
Recall:0.460
fbeta:0.599


## Ethical Considerations

The data may include sensitive data, although it is anonymized.
