Toy example of using Xgboost and automl to predict MEDV in Boston dataset.
My personal preference is automl due to the convenience and simple code.

Special Note for automl==2.9.9
I needed to change the line 10 in DataFrameVectorizer.py because sklearn.externals no longer has six.
The change is
from: from sklearn.external import six 
to: import six