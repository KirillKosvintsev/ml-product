from typing import Union, Literal, Optional, Tuple, List

from pydantic import BaseModel, validator

N_JOBS = 4


class DecisionTreeRegressorParameters(BaseModel):  # ready
    criterion: Literal['squared_error', 'friedman_mse', 'absolute_error', 'poisson'] = 'squared_error'
    splitter: Literal['best', 'random'] = 'best'
    max_depth: int = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1

    @validator("max_depth")
    def validate_max_depth(cls, value):
        if value <= 0:
            raise ValueError('max_depth must be greater than zero.')
        return value

    @validator("min_samples_split")
    def validate_min_samples_split(cls, value):
        if value < 2:
            raise ValueError('min_samples_split must be 2 or greater.')
        return value

    @validator("min_samples_leaf")
    def validate_min_samples_leaf(cls, value):
        if value < 1:
            raise ValueError('min_samples_leaf must be 1 or greater.')
        return value


class AdaBoostRegressorParameters(BaseModel):
    loss: Literal['linear', 'square', 'exponential'] = 'linear'
    learning_rate: float = 1.0
    n_estimators: int = 50

    @validator("learning_rate")
    def validate_learning_rate(cls, value):
        if value <= 0:
            raise ValueError('learning_rate must be greater than zero.')
        return value

    @validator("n_estimators")
    def validate_n_estimators(cls, value):
        if value < 2:
            raise ValueError('n_estimators must be 2 or greater.')
        return value

class KNeighborsRegressorParameters(BaseModel):
    n_neighbors : int = 5
    weights: Literal['uniform', 'distance'] = 'uniform'
    algorithm : Literal['auto', 'ball_tree', 'kd_tree', 'brute'] = 'auto'
    leaf_size: int = 30
    p: int = 2
    n_jobs: Optional[int] = N_JOBS


    @validator("n_neighbors")
    def validate_n_neighbors(cls, value):
        if value <= 0:
            raise ValueError('n_neighbors must be greater than zero.')
        return value

    @validator("leaf_size")
    def validate_leaf_size(cls, value):
        if value < 0:
            raise ValueError('leaf_size must be 1 or greater.')
        return value

    @validator("p")
    def validate_p(cls, value):
        if value < 0:
            raise ValueError('p must be 1 or greater.')
        return value

class LinearSVRParameters(BaseModel):
    epsilon: float = 0.0
    tol: float = 1e-4
    C: float = 1.0
    loss: Literal['epsilon_insensitive', 'squared_epsilon_insensitive'] = 'epsilon_insensitive'
    fit_intercept: bool = True
    dual: bool = True


    @validator("epsilon")
    def validate_epsilon(cls, value):
        if value < 0:
            raise ValueError('epsilon must be 0 or greater.')
        return value

    @validator("tol")
    def validate_tol(cls, value):
        if value < 0:
            raise ValueError('tol must be greater than zero.')
        return value

    @validator("C")
    def validate_C(cls, value):
        if value < 0:
            raise ValueError('C must be than zero.')
        return value

class SVRParameters(BaseModel):
    epsilon: float = 0.0
    tol: float = 1e-4
    C: float = 1.0
    kernel: Literal['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'] = 'rbf'
    degree: int = 3
    gamma: Literal['scale', 'auto'] = 'scale'

    @validator("epsilon")
    def validate_epsilon(cls, value):
        if value < 0:
            raise ValueError('epsilon must be 0 or greater.')
        return value

    @validator("tol")
    def validate_tol(cls, value):
        if value < 0:
            raise ValueError('tol must be greater than zero.')
        return value

    @validator("C")
    def validate_C(cls, value):
        if value < 0:
            raise ValueError('C must be greater than zero.')
        return value

    @validator("degree")
    def validate_degree(cls, value):
        if value < 0:
            raise ValueError('degree must be 0 or greater.')
        return value

class MLPRegressorParameters(BaseModel):
    hidden_layer_sizes: List[int] = [100]
    activation : Literal['identity', 'logistic', 'tanh', 'relu'] = 'relu'
    solver : Literal['lbfgs', 'sgd', 'adam'] = 'adam'
    alpha: float = 0.0001
    learning_rate : Literal['constant', 'invscaling', 'adaptive'] = 'constant'
    learning_rate_init: float = 0.001
    max_iter: int = 1000
    shuffle: bool = True
    n_iter_no_change: int = 10
    tol: float = 1e-4


    @validator("max_iter")
    def validate_max_iter(cls, value):
        if value <= 0:
            raise ValueError('max_iter must be greater than zero.')
        return value

    @validator("alpha")
    def validate_alpha(cls, value):
        if value < 0:
            raise ValueError('alpha must be 0 or greater.')
        return value

    @validator("tol")
    def validate_tol(cls, value):
        if value < 0:
            raise ValueError('tol must be greater than zero.')
        return value

    @validator("learning_rate_init")
    def validate_learning_rate_init(cls, value):
        if value < 0:
            raise ValueError('learning_rate_init must be greater than zero.')
        return value

    @validator("n_iter_no_change")
    def validate_n_iter_no_change(cls, value):
        if value < 1:
            raise ValueError('n_iter_no_change must be 1 or greater.')
        return value
