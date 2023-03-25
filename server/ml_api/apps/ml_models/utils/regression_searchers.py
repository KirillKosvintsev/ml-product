from hyperopt import hp


REGRESSION_CONFIG = {
    'DecisionTreeRegressor': {
        'criterion': hp.choice(label='criterion', options=['gini', 'entropy', 'squared_error', 'friedman_mse', 'absolute_error', 'poisson']),
        'max_depth': hp.randint(label='max_depth', low=3, high=30),
    },
    'BaggingRegressor': {
        'n_estimators': hp.randint(label="n_estimators", low=10, high=100),
    }
}
