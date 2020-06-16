from auto_ml import Predictor
from auto_ml.utils import get_boston_dataset

class automl_tutorial():
    def __init__(self):
        self.df_train, self.df_test = None, None
        self.column_descriptions = {
            'MEDV': 'output',
            'CHAS': 'categorical'
        }
        self.ml_predictor = None

    def run(self):
        self.load_data()
        self.fit_predict()

    def load_data(self):
        self.df_train, self.df_test = get_boston_dataset()

    def fit_predict(self):
        self.ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=self.column_descriptions)
        self.ml_predictor.train(self.df_train)
        self.ml_predictor.score(self.df_test, self.df_test.MEDV)

if __name__ == '__main__':
    a = automl_tutorial()
    a.run()