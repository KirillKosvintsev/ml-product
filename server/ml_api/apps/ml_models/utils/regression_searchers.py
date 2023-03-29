from hyperopt import hp

REGRESSION_CONFIG = {
    'DecisionTreeRegressor': {
        'criterion': hp.choice(label='criterion',
                               options=['gini', 'entropy', 'squared_error',
                                        'friedman_mse', 'absolute_error',
                                        'poisson']),
        'max_depth': hp.randint(label='max_depth', low=3, high=30),
    },
    'AdaBoostRegressor': {
        'loss': hp.choice(label='loss',
                          options=['linear', 'square', 'exponential']),
        'learning_rate': hp.uniform('learning_rate', 1e-6, 1),
        'n_estimators': hp.randint(label="n_estimators", low=50, high=300)
    },
    'KNeighborsRegressor': {
        'n_neighbors': hp.randint(label="n_neighbors", low=1, high=10),
        'weights': hp.choice(label='weights', options=['uniform', 'distance']),
        'algorithm': hp.choice(label='algorithm',
                               options=['auto', 'ball_tree', 'kd_tree',
                                        'brute']),
        'p': hp.randint(label="p", low=1, high=10)
    },
    'LinearSVR': {
        'loss': hp.choice(label='loss', options=['epsilon_insensitive',
                                                 'squared_epsilon_insensitive']),
        'epsilon': hp.uniform('epsilon', 0, 5),
        'C': hp.uniform('C', 0, 10)
    },
    'SVR': {
        'kernel': hp.choice(label='kernel',
                            options=['linear', 'poly', 'rbf', 'sigmoid',
                                     'precomputed']),
        'gamma': hp.choice(label='gamma', options=['scale', 'auto']),
        'epsilon': hp.uniform('epsilon', 0, 5),
        'C': hp.uniform('C', 0, 10),
        'degree': hp.randint(label="degree", low=0, high=10)
    },
    'MLPRegressor': {
        'activation': hp.choice(label='activation',
                                options=['identity', 'logistic', 'tanh',
                                         'relu']),
        'solver': hp.choice(label='solver', options=['lbfgs', 'sgd', 'adam']),
        'learning_rate': hp.choice(label='learning_rate',
                                   options=['constant', 'invscaling',
                                            'adaptive']),
        'alpha': hp.uniform('alpha', 0, 1),
        'learning_rate_init': hp.uniform('learning_rate_init', 0, 1),
    },
    'BaggingRegressor': {
        'n_estimators': hp.randint(label="n_estimators", low=10, high=100),
    },
    'ExtraTreeRegressor': {
        'n_estimators': hp.randint(label="n_estimators", low=50, high=300),
        'criterion': hp.choice(label='criterion',
                               options=['squared_error', 'absolute_error',
                                        'friedman_mse', 'poisson']),
        'max_depth': hp.randint(label='max_depth', low=3, high=20),
    },
    'GradientBoostringRegressor': {
        'loss': hp.choice(label='loss',
                          options=['squared_error', 'absolute_error', 'huber',
                                   'quantile']),
        'learning_rate': hp.uniform('learning_rate', 1e-6, 1),
        'n_estimators': hp.randint(label="n_estimators", low=50, high=300),
        'criterion': hp.choice(label='criterion',
                               options=['friedman_mse', 'squared_error']),
        'max_depth': hp.randint('max_depth', 3, 20),
    },
    'RandomForestRegressor': {
        'n_estimators': hp.randint(label='n_estimators', low=50, high=300),
        'criterion': hp.choice(label='criterion',
                               options=['squared_error', 'absolute_error',
                                        'friedman_mse', 'poisson']),
        'max_depth': hp.randint(label='max_depth', low=3, high=20),
    },
    'LinearRegression': {
    },
    'SGDRegressor': {
        'loss': hp.choice(label='loss', options=['squared_error', 'huber',
                                                 'epsilon_insensitive',
                                                 'squared_epsilon_insensitive']),
        'penalty': hp.choice(label='penalty',
                             options=['l2', 'l1', 'elasticnet', 'none']),
        'alpha': hp.uniform('alpha', 1e-6, 1e-3),
        'l1_ratio': hp.uniform('l1_ratio', 0.05, 0.95),
        'learning_rate': hp.choice(label='learning_rate',
                                   options=['constant', 'optimal',
                                            'invscaling', 'adaptive']),
    },
}
