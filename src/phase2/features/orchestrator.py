import pickle
import pandas as pd
from loguru import logger
# from src.phase2 import cython_code
import cython_code


class Orchestrator:
    def __init__(self):
        self.columns = ['feature{}'.format(i) for i in range(1, 17)]
        with open('./models/pipeline_1.pkl', 'rb') as f:
            self.pipeline1 = pickle.load(f)

        with open('./models/pipeline_2.pkl', 'rb') as f:
            self.pipeline2 = pickle.load(f)

        with open('./models/problem1.pkl', 'rb') as g:
            self.model1 = pickle.load(g)

        with open('./models/problem2.pkl', 'rb') as g:
            self.model2 = pickle.load(g)

    def transform(self, X, model):
        if model == 'prob1':
            return self.pipeline1.transform(X)
        else:
            return self.pipeline2.transform(X)

    def predict(self, data, model):
        logger.info("Transform")
        data = self.transform(data, model)
        logger.info("Predict")

        if model == 'prob1':
            return cython_code.predict_proba_catboost(data, self.model1)
        else:
            return cython_code.predict_catboost(data, self.model2)
