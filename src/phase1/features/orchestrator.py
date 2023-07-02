import pickle
import pandas as pd
import numpy as np


class Orchestrator:
    def __init__(self):
        self.columns = ['feature{}'.format(i) for i in range(1, 17)]
        with open('./models/pipeline.pkl', 'rb') as f:
            self.pipeline1 = pickle.load(f)

        with open('./models/pipeline_2.pkl', 'rb') as f:
            self.pipeline2 = pickle.load(f)

        with open('./models/problem1.pkl', 'rb') as g:
            self.model1 = pickle.load(g)

        with open('./models/problem2.pkl', 'rb') as g:
            self.model2 = pickle.load(g)

        self.columns = ["feature{}".format(i) for i in range(1, 17)]

    def transform(self, X, columns, model):
        df = pd.DataFrame(X, columns=columns)
        if model == 'prob1':
            return self.pipeline1.transform(df)
        else:
            return self.pipeline2.transform(df)

    def predict(self, data, columns, model):
        data = self.transform(data, columns, model)
        if model == 'prob1':
            prob = self.model1.predict_proba(data)[:, 1]
        else:
            prob = self.model2.predict_proba(data)[:, 1]
        return prob


a = [-1, 1.65e-05, 3.19e-05, 5.47e-05, 9.29e-05, 0.000169, 0.000326, 0.000726, 0.00225, 0.022, 1.1]
b = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
cols_psi = list('ABCDEFGHIK')


def cal_psi(inferences):
    c = list(pd.cut(inferences, a, labels=cols_psi))
    psi = 0
    for col in cols_psi:
        tu = c.count(col) / len(c) + 0.0000000001
        psi += (tu - 0.1) / np.log(tu / 0.1)
    return psi