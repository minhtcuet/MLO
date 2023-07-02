import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from optbinning import BinningProcess, OptimalBinning
from joblib import Parallel, delayed


import warnings

warnings.filterwarnings("ignore")


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, _feature_names):
        self._feature_names = _feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self._feature_names]


class WOE(BaseEstimator, TransformerMixin):
    def __init__(self, cats, nums):
        self.cats = cats
        self.nums = nums
        self.res = {}

    def fit(self, X, y):
        for col in self.cats:
            optb = OptimalBinning(name=col, dtype="categorical", solver="cp")
            optb.fit(X[col].values, y.values)
            self.res[col] = optb

        for col in self.nums:
            optb = OptimalBinning(name=col, dtype="numerical", solver="cp")
            optb.fit(X[col].values, y.values)
            self.res[col] = optb
        return self

    # def transform(self, X, y=None):
    #     for col in self.cats:
    #         X[col] = self.res[col].transform(X[col], metric='woe')
    #     return X

    def _transform_column(self, col, X):
        return self.res[col].transform(X[col], metric='woe')

    def transform(self, X, y=None):
        transformed_columns = Parallel(n_jobs=-1)(
            delayed(self._transform_column)(col, X) for col in self.cats
        )

        for col, transformed_column in zip(self.cats, transformed_columns):
            X[col] = transformed_column

        return X
