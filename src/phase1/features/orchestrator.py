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


a = [-1, 1.05848934e-06, 3.02380710e-06, 7.11921902e-06,
     1.56224858e-05, 3.33687196e-05, 7.61984550e-05, 1.99051388e-04,
     6.99545860e-04, 7.59180369e-03, 1.1]
b = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
cols_psi = list('ABCDEFGHIK')


def cal_psi_pro1(inferences):
    try:
        vc = list(pd.cut(inferences, a, labels=cols_psi))
        psi = 0
        for col in cols_psi:
            tu = vc.count(col) / len(vc) + 0.0000000001
            psi += (tu - 0.1) / np.log(tu / 0.1)
        return psi
    except:
        return 1


c = [-1, 2.56203554e-02, 5.11177520e-02, 8.45635855e-02,
     1.31936916e-01, 1.98857889e-01, 2.94749266e-01, 4.25216823e-01,
     5.86057397e-01, 7.66938595e-01, 1.1]
d = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]


def cal_psi_pro2(inferences):
    try:
        e = list(pd.cut(inferences, c, labels=cols_psi))
        psi = 0
        for col in cols_psi:
            tu = e.count(col) / len(e) + 0.0000000001
            psi += (tu - 0.1) / np.log(tu / 0.1)
        return psi
    except:
        return 1
