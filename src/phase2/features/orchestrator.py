import pickle
import pandas as pd


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

    def transform(self, X, columns, model):
        df = pd.DataFrame(X, columns=columns)
        if model == 'prob1':
            return self.pipeline1.transform(df)
        else:
            return self.pipeline2.transform(df)

    def predict(self, data, columns, model):
        data = self.transform(data, columns, model)
        if model == 'prob1':
            return list(self.model1.predict_proba(data)[:, 1])
        else:
            return list(self.model2.predict(data)[:, 0].tolist())
