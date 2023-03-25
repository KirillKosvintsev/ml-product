from typing import Union, Literal, Optional, Tuple

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

class BaggingRegressorParameters(BaseModel):
    n_estimators: int = 10
    max_samples: Union[int, float] = 1
    max_features: Union[int, float] = 1
    bootstrap: bool = True
    bootstrap_features: bool = False
    n_jobs: Optional[int] = N_JOBS 

    @validator("n_estimators")
    def validate_n_estimators(cls, value):
        if value < 2:
            raise ValueError('n_estimators must be 2 or greater.')
        return value

    @validator("max_samples")
    def validate_max_samples(cls, value):
        if value < 1:
            raise ValueError('max_samples must be 1 or greater.')
        return value

    @validator("max_features")
    def validate_max_features(cls, value):
        if value < 1:
            raise ValueError('max_features must be 1 or greater.')
        return value

class ExtraTreeRegressorParameters(BaseModel):
    n_estimators: int = 100
    criterion: Literal['squared_error', 'absolute_error', 'friedman_mse', 'poisson'] = 'squared_error'
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    bootstrap: bool = False
    n_jobs: Optional[int] = N_JOBS 

    @validator("n_estimators")
    def validate_n_estimators(cls, value):
        if value < 2:
            raise ValueError('n_estimators must be 2 or greater.')
        return value

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

class GradientBoostingRegressorParameters(BaseModel):
    loss: Literal['squared_error', 'absolute_error', 'huber', 'quantile'] = 'squared_error'
    learning_rate: float = 0.1
    n_estimators: int = 100
    subsample: float = 1
    criterion: Literal['friedman_mse', 'squared_error'] = 'friedman_mse'
    min_samples_split: int = 2
    max_depth: int = 3

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

    @validator("subsample")
    def validate_subsample(cls, value):
        if value > 1 or value <= 0:
            raise ValueError("subsample must be in (0, 1]")
        return value

    @validator("min_samples_split")
    def validate_min_samples_split(cls, value):
        if value < 2:
            raise ValueError('min_samples_split must be 2 or greater.')
        return value

    @validator("max_depth")
    def validate_max_depth(cls, value):
        if value <= 0:
            raise ValueError('max_depth must be greater than zero.')
        return value

class RandomForestRegressorParameters(BaseModel):
    n_estimators: int = 100
    criterion: Literal['squared_error', 'absolute_error', 'friedman_mse', 'poisson'] = 'squared_error'
    max_depth: int = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    bootstrap: bool = True

    @validator("n_estimators")
    def validate_n_estimators(cls, value):
        if value < 2:
            raise ValueError('n_estimators must be 2 or greater.')
        return value

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

class LinearRegressionParameters(BaseModel):
    fit_intercept: bool = True
    copy_X: bool = True
    n_jobs: Optional[int] = N_JOBS
    positive: bool = False

class SGDRegressorParameters(BaseModel):
    loss: Literal['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'] = 'hinge'
    penalty: Optional[Literal['l2', 'l1', 'elasticnet']] = 'l2'
    alpha: float = 0.0001
    l1_ratio: float = 0.15
    fit_intercept: bool = True
    max_iter: int = 1000
    shuffle: bool = True
    epsilon: float = 0.1
    n_jobs: Optional[int] = N_JOBS
    learning_rate: Literal['constant', 'optimal', 'invscaling', 'adaptive'] = 'invscaling'

    @validator("alpha")
    def validate_alpha(cls, value):
        if value < 0:
            raise ValueError('alpha must be non-negative.')
        return value

    @validator("l1_ratio")
    def validate_l1_ratio(cls, value):
        if value < 0 or value > 1:
            raise ValueError('l1_ratio must be in [0, 1].')
        return value

    @validator("max_iter")
    def validate_max_iter(cls, value):
        if value <= 0:
            raise ValueError('max_iter must be greater than zero.')
        return value

    @validator("epsilon")
    def validate_epsilon(cls, value):
        if value < 0:
            raise ValueError('epsilon must be non-negative.')
        return value
