VARIABLES_COLUMNS_REMOVE_IN_STEP1_REDUNDANCE = []

import pandas as pd 
import numpy as np 
from datetime import datetime 
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn import metrics
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
import warnings
from optbinning import BinningProcess
from optbinning import OptimalBinning, MulticlassOptimalBinning
from sklearn.compose import ColumnTransformer
from sklearn import set_config
import re
from sklearn.feature_selection import VarianceThreshold

class MissingRemoving(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self._feature_names = None
        self.removes = []
        self.tmp = None
    
    def fit(self, X, y=None):
        fil = X.isnull().sum(axis = 0) / X.shape[0]
        self.tmp = fil
        self._feature_names = fil[fil <= self.threshold].index
        self.removes = fil[fil > self.threshold].index
        global VARIABLES_COLUMNS_REMOVE_IN_STEP1_REDUNDANCE
        VARIABLES_COLUMNS_REMOVE_IN_STEP1_REDUNDANCE += list(self.removes)
        return self

    def transform(self, X, y=None):
        return X[self._feature_names]

class DuplicatedReduction(BaseEstimator, TransformerMixin):
    def __init__(self, nums):
        self.dup = []
        self.low_freq = []
        self.removes = []
        self.nums = nums
        
    def duplicate_columns(self, X):
        self.dup = [c for c, flag in X.T.duplicated().items() if flag]
        return self.dup
    
    def low_frequency(self, X):
        vt = VarianceThreshold(threshold=0)
        vt.fit(X)
        
        self.low_freq = [X.columns[c] for c, flag in enumerate(vt.get_support()) if not flag]
        return self.low_freq
    
    def fit(self, X, y=None):
        self.duplicate_columns(X)
        self.low_frequency(X[self.nums])
        self.removes = self.dup + self.low_freq
        global VARIABLES_COLUMNS_REMOVE_IN_STEP1_REDUNDANCE
        VARIABLES_COLUMNS_REMOVE_IN_STEP1_REDUNDANCE += self.removes
        return self
    
    def transform(self, X, y=None):
        X = X.drop(columns=self.removes)
        remaining_cols = list(X.columns)
        return X[remaining_cols]

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, _feature_names):
        self._feature_names = _feature_names

    def fit(self, X, y=None):
        self._feature_names = [col for col in self._feature_names if col not in VARIABLES_COLUMNS_REMOVE_IN_STEP1_REDUNDANCE]
        return self

    def transform(self, X, y=None):
        return X[self._feature_names]

class ImputeNegative(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        self.features = [col for col in self.features if col not in VARIABLES_COLUMNS_REMOVE_IN_STEP1_REDUNDANCE]
        return self

    def transform(self, X, y=None):
        for col in self.features:
            X[col] = np.where(X[col] < 0, None, X[col])
        return X

class RedundanceRemoving(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self._feature_names = feature_names
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X.drop(columns=self._feature_names)

class OutlierAndRedundanceCatDetector(BaseEstimator, TransformerMixin):
    def __init__(self, cats):
        self.cats = cats
        self.replacing = replacing
        
    def fit(self, X, y=None):
        self.cats = [col for col in self.cats if col not in VARIABLES_COLUMNS_REMOVE_IN_STEP1_REDUNDANCE]
        return self
    
    def transform(self, X, y=None):
        for key, val in self.replacing.items():
            if key == 'TEMPORARYADDRESSPROVINCE' or key not in self.cats:
                continue
            if key == 'TYPEOFLABOURCONTRACT':
                X[key] = X[key].map(val['value'])
                continue
            X[key] = X[key].map(val['value']).fillna(val['mode'])
        return X

class OutlierDetector(BaseEstimator, TransformerMixin):
    def __init__(self, feas, drop=False, 
                 replace_quantile=True, 
                 threshold=0.99, 
                 inference=False):
        self.feas = feas
        self.drop = drop
        self.replace_quantile = replace_quantile
        self.threshold = threshold
        self.inference = inference
        if self.drop == self.replace_quantile == True:
            raise Exception("Only choose replace_quantile or drop in OutlierDetector")
        self.outlier = {}
        self.feature_name = list()
        
    def fit(self, X, y=None):
        self.feas = [col for col in self.feas if col not in VARIABLES_COLUMNS_REMOVE_IN_STEP1_REDUNDANCE]
        self.outlier_detect(X)
        return self
    
    def outlier_detect(self, X):
        for col in self.feas:
            self.outlier[col] = X[col].quantile(0.99)
    
    def get_feature_names(self):
        return list(self.feature_name)
        
    def transform(self, X, y=None):
        if self.replace_quantile:
            for col in self.feas:
                X[col] = np.where(X[col] >= self.outlier[col], self.outlier[col], X[col])
        if self.drop:
            for col in X.columns:
                X = X[X[col] <= self.outlier[col]]
        self.feature_name = list(X.columns)
        return X
    

from scipy import stats

class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, epsilon=1e-6):
        self.columns = columns
        self.epsilon = epsilon

    def fit(self, X, y=None):
        if self.columns is None:
            raise ValueError("You must provide a list of column names to apply the Box-Cox transformation.")
        
        self.columns = [col for col in self.columns if col not in VARIABLES_COLUMNS_REMOVE_IN_STEP1_REDUNDANCE]
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for column in self.columns:
            not_na_rows = X_transformed[column].notna()
            X_transformed.loc[not_na_rows, column] = X_transformed.loc[not_na_rows, column].apply(lambda x: np.log(x + self.epsilon))
        return X_transformed


class ProvinceConvert(BaseEstimator, TransformerMixin):
    def __init__(self, _feature_name):
        self.feature_name = list()
        self._feature_name = _feature_name
        self.flag = False
    
    def fit(self, X, y=None):
        self.flag = self._feature_name in VARIABLES_COLUMNS_REMOVE_IN_STEP1_REDUNDANCE
        return self
    
    def get_feature_names(self):
        return list(self.feature_name)
    
    def transform(self, X, y=None):
        if not self.flag:
            X[self._feature_name] = X[self._feature_name].map(replacing[self._feature_name]['value'])
            self.feature_name = X.columns
            return X
        return X

VARIABLE_STORAGE_IV = {}

class WoeImputing(BaseEstimator, TransformerMixin):
    def __init__(self, training=True ,inference=False, cats=list(), nums=list()):
        self._feature_names = list()
        self.inference = inference
        self.training = training
        self.cats = cats
        self.nums = nums
        self._validate_input()
        self.mem = dict()
    
    def fit(self, X, y):
        samples = [i for i in list(self.nums) + list(self.cats) if i in X.columns]
        global VARIABLE_STORAGE_IV
        for col in samples:
            if col in self.cats:
                optb = MulticlassOptimalBinning(name=col, solver="cp")
            else:
                optb = MulticlassOptimalBinning(name=col, solver="cp", min_n_bins=5)
            optb.fit(X[col].values, y.values)
            self.mem[col] = optb

        return self
    
    def _validate_input(self):
        if self.training == self.inference == True:
            raise Exception("Only Training or Inference in WoeImputing")
        if self.training == self.inference == False:
            raise Exception("At least a parameter training or inference need to be set")
    
    def _round(self, number):
        return int(number * 10 ** 8) / 10 ** 8
        
    def transform(self, X, y=None):
        samples = [i for i in list(self.nums) + list(self.cats) if i in X.columns]
        for col in samples:
            if col in self.cats:
                X[col] = self.mem[col].transform(X[col], metric="event_rate")
            else:
                X[col] = self.mem[col].transform(X[col], metric="woe")
            X[col] = X[col].apply(self._round)
        return X
        

class RemoveHighCorr(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.couples = []
        self.remove = []
    
    def create_couple(self, X):
        vl = X.corr()
        for col in vl.index:
            for co in vl.columns:
                if col == co:
                    continue
                corr = vl.loc[col, co]
                if abs(corr) > self.threshold:
                    self.couples.append([sorted((col, co)), vl.loc[col, co]])
        
    def remove_corr(self):
        while self.couples:
            self.couples = sorted(self.couples, key=lambda x: x[1])[::-1]
            couple = self.couples.pop(0) 
            a, b = couple[0]
            min_ = sorted([(a, VARIABLE_STORAGE_IV[a]), (b, VARIABLE_STORAGE_IV[b])], key=lambda x: x[1])[0][0]
            self.remove.append(min_)
            self.couples = [couple for couple in self.couples if min_ not in couple[0]]
        return self.remove
            
    def fit(self, X, y=None):
        self.create_couple(X)
        out = self.remove_corr()
        return self
    
    def transform(self, X, y=None):
        X = X.drop(columns=self.remove)
        return X

class ImputeRatio(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features
    
    def fit(self, X, y=None):
        self.features = [i for i in self.features if i not in VARIABLES_COLUMNS_REMOVE_IN_STEP1_REDUNDANCE]
        return self 
    
    def transform(self, X, y=None):
        for col in self.features:
            if col not in X:
                continue
            X[col] = X[col].apply(lambda x: 'XX' if x > 1.5 else x)
        return X


class PandasFeatureUnion(FeatureUnion):
    def __init__(self, transformer_list, n_jobs=None, transformer_weights=None, verbose=False):
        super().__init__(transformer_list, n_jobs=n_jobs, transformer_weights=transformer_weights, verbose=verbose)
        
    def feature_union_transform(self, X):
        transformed = [transformer.transform(X) for _, transformer in self.transformer_list]
        X_transformed = pd.concat([pd.DataFrame(t) for t in transformed], axis=1)
        return X_transformed
        
    def transform(self, X):
        return self.feature_union_transform(X)
    
    
