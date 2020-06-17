Toy example of using Xgboost and automl to predict MEDV in Boston dataset.
My personal preference is automl due to the convenience and simple code.

Scikit-Opt Results
Scikit-Opt is a powerful tool that improved F1 score in example by 13.2% (from 76.3% to 89.5%). On top of that, it is convenient and easy to use, with very clean syntax to define search space.

Special Note for automl==2.9.9
I needed to change the line 10 in DataFrameVectorizer.py because sklearn.externals no longer has six.
The change is
from: from sklearn.external import six 
to: import six