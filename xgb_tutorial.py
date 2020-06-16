# as learned on https://www.datacamp.com/community/tutorials/xgboost-in-python
# use regressor to predict property price

import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class xgb_class():
    def __init__(self):
        self.data = None
        self.X, self.Y = None, None
        self.X_train, self.Y_train, self.X_test, self.Y_test = None, None, None, None
        self.xg_regressor = None
        self.preds = None
        self.rmse = None
        self.cv_results = None
        self.Dmatrix = None
        self.params = {"objective": "reg:squarederror", 'colsample_bytree': 0.3, 'learning_rate': 0.1,
                       'max_depth': 5, 'alpha': 10}

    def run(self):
        self.prepare_data()
        self.fit_predict()
        self.make_Dmatrix()
        self.cross_validation()
        self.viz_trees()
        self.plot_importance()

    def fit_predict(self):
        self.xg_regressor = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                                  max_depth=5, alpha=10, n_estimators=10)
        self.xg_regressor.fit(self.X_train, self.Y_train)
        self.preds = self.xg_regressor.predict(self.X_test)
        self.rmse = np.sqrt(mean_squared_error(self.Y_test, self.preds))
        print(self.rmse)

    def make_Dmatrix(self):
        self.Dmatrix = xgb.DMatrix(data=self.X,label=self.Y)

    def cross_validation(self):
        self.cv_results = xgb.cv(dtrain=self.Dmatrix, params=self.params, nfold=3,
                            num_boost_round=50, early_stopping_rounds=10, metrics="rmse", as_pandas=True, seed=123)
        print(self.cv_results.head())
        print(self.cv_results['test-rmse-mean'].tail(1))

    def viz_trees(self):
        self.xg_regressor = xgb.train(params=self.params, dtrain=self.Dmatrix, num_boost_round=10)
        xgb.plot_tree(self.xg_regressor, num_trees=0)
        plt.rcParams['figure.figsize'] = [50, 10]
        plt.show()

    def plot_importance(self):
        xgb.plot_importance(self.xg_regressor)
        plt.rcParams['figure.figsize'] = [5, 5]
        plt.show()

    def prepare_data(self):
        self.load_data()
        self.train_test_sets()

    def load_data(self):
        _ = load_boston()
        self.data = pd.DataFrame(_.data)
        self.data.columns = _.feature_names
        self.data['Price'] = _.target
        self.X, self.Y = self.data.iloc[:,:-1], self.data.iloc[:,-1]

    def explore_data(self):
        print(self.X.info())
        print('\n')
        print(self.X.describe())
        print('\n')
        print(self.X.shape)
        print('\n')
        print(self.X.head())

    def train_test_sets(self, test_size=None, random_state=None):
        if test_size is None: test_size = 0.2
        if random_state is None: random_state = 0
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=test_size, random_state=random_state)

if __name__ == '__main__':
    x = xgb_class()
    x.run()