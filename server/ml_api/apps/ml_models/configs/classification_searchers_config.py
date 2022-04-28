from hyperopt import hp
import numpy as np


CLASSIFICATION_SEARCHERS_CONFIG = {
    'DecisionTreeClassifier': {
        'criterion': hp.choice(label='criterion', options=['gini', 'entropy']),
        'max_depth': hp.randint(label='max_depth', low=2, high=30),
    },
    'CatBoostClassifier': {
        'learning_rate':     hp.uniform('learning_rate', 0.001, 0.1),
        'max_depth':         hp.randint('max_depth', 5, 16),
        'colsample_bylevel': hp.uniform('colsample_bylevel', 0.3, 0.8)
    }
}