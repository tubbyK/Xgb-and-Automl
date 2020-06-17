# as learned on:
# https://towardsdatascience.com/how-to-improve-the-performance-of-xgboost-models-1af3995df8ad
# https://scikit-optimize.github.io/stable/auto_examples/sklearn-gridsearchcv-replacement.html
# https://towardsdatascience.com/hyperparameter-optimization-in-python-part-1-scikit-optimize-754e485d24fe

from xgboost import XGBClassifier
from skopt import gp_minimize
from skopt.space import Real, Integer
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from functools import partial
import time

class xgb_auto_opt():
    def __init__(self, seed=None, n_samples=None, n_features=None, n_informative=None, n_classes=None,
                 class_weights=None, test_ratio=None, n_calls =None):
        self.start = time.time()
        self.end = None
        self.seed = 42 if seed is None else seed
        np.random.seed(self.seed)
        self.n_samples = 10_000 if n_samples is None else n_samples
        self.n_features = 30 if n_features is None else n_features
        self.n_informative = 10 if n_informative is None else n_informative
        self.n_classes = 2 if n_classes is None else n_classes
        self.class_weights = 0.6 if class_weights is None else class_weights
        self.test_ratio = 0.2 if test_ratio is None else test_ratio
        self.n_calls = 100 if n_calls  is None else n_calls
        self.df_train, self.df_test = None, None
        self.X, self.y = None, None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.space = None
        self.models, self.train_scores, self.test_scores = [], [], []
        self.results = None
        self.func = 1.0

    def run(self):
        self.load_data()
        self.define_space()
        objective_function = partial(self.return_model_assessment)
        self.results = gp_minimize(objective_function, self.space, base_estimator=None, n_calls=self.n_calls ,
                              n_random_starts=self.n_calls  - 1, random_state=self.seed, callback=[self.monitor])
        print(self.results)
        self.end = time.time()
        print(f'time taken: {(self.end - self.start)/60:.3}')

    def load_data(self):
        self.X, self.y = make_classification(n_samples=self.n_samples, n_features=self.n_features,
                                             n_informative=self.n_informative, n_classes=self.n_classes,
                                             weights=[self.class_weights], random_state=self.seed)
        col_names = ['col_' + str(i + 1) for i in range(self.X.shape[1])]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, stratify=self.y,
                                                                                test_size=self.test_ratio,
                                                                                random_state=self.seed)
        target_col = "target"
        self.df_train = pd.DataFrame(self.X_train, columns=col_names)
        self.df_train.loc[:, target_col] = self.y_train
        self.df_test = pd.DataFrame(self.X_test, columns=col_names)
        self.df_test.loc[:, target_col] = self.y_test

    def define_space(self):
        self.space = [
            Real(0.01, 1.0, name="colsample_bylevel", prior='log-uniform'),
            Real(0.01, 1.0, name="colsample_bytree", prior='log-uniform'),
            Real(0.01, 1, name="gamma", prior='log-uniform'),
            Real(0.00001, 1, name="learning_rate", prior='log-uniform'),
            Real(0.01, 10, name="max_delta_step", prior='log-uniform'),
            Integer(1, 30, name="max_depth"),
            Real(0.01, 5_000, name="min_child_weight", prior='log-uniform'),
            Integer(3, 10_000, name="n_estimators"),
            Real(0.01, 100, name="reg_alpha", prior='log-uniform'),
            Real(0.01, 100, name="reg_lambda", prior='log-uniform'),
            Real(0.01, 1.0, name="subsample", prior='log-uniform'),
        ]

    def monitor(self, res):
        if len(res.func_vals)>1:
            if res.func_vals[-1] <= self.func:
                self.func = res.func_vals[-1]
                print('iteration', len(res.func_vals))
                print('run_score', 1 - res.func_vals[-1])
                print('run_parameters', (res.x_iters[-1]))

    def return_model_assessment(self, args):
        curr_model_hyper_params = ['colsample_bylevel', 'colsample_bytree', 'gamma', 'learning_rate', 'max_delta_step',
                                   'max_depth', 'min_child_weight', 'n_estimators', 'reg_alpha', 'reg_lambda',
                                   'subsample']
        params = {curr_model_hyper_params[i]: args[i] for i, j in enumerate(curr_model_hyper_params)}
        model = XGBClassifier(random_state=self.seed, seed=self.seed)
        model.set_params(**params)
        fitted_model = model.fit(self.X_train, self.y_train, sample_weight=None)
        self.models.append(fitted_model)
        train_predictions = model.predict(self.X_train)
        test_predictions = model.predict(self.X_test)
        train_score = f1_score(train_predictions, self.y_train)
        test_score = f1_score(test_predictions, self.y_test)
        self.train_scores.append(train_score)
        self.test_scores.append(test_score)
        return 1 - test_score

if __name__ == '__main__':
    x = xgb_auto_opt()
    x.run()