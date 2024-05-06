import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from IPython.core.display_functions import display
from seaborn import barplot
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
import time

# x_train, x_test, y_train, y_test = train_test_split(0.20)
# data = pandas.DataFrame(np.arange(12).reshape((4,3)), columns=['a', 'b', 'c'])
# label = pandas.DataFrame(np.random.randint(2, size=4))
# dtrain = xgb.DMatrix(data, label=label)
#
# dtrain = xgb.DMatrix('train.svm.txt')
# dtrain.save_binary('train.buffer')
#
# # label_column specifies the index of the column containing the true label
# dtrain = xgb.DMatrix('train.csv?format=csv&label_column=')
# dtest = xgb.DMatrix('test.csv?format=csv&label_column=0')
# multi:softmax: set XGBoost to do multiclass classification using the softmax objective, you also need to set
# num_class(number of classes)


from xgboost import XGBClassifier
# read data
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def ml():
    data = pd.read_csv("CSVs/eyetracking_by_event.csv")

    X, y = data.loc[:, ~data.columns.isin(['Thought_Type', 'Participant'])], data['Thought_Type']

    imputer = KNNImputer(n_neighbors=2, weights='uniform')
    X = imputer.fit_transform(X, y)

    smote = SMOTE()
    X, y = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

    parameters = {'n_estimators': [1000],
                  'max_depth': [4],
                  'learning_rate': [0.001]}

    best_params = {'learning_rate': 0.001, 'max_depth': 4, 'n_estimators': 1000}

    # create model instance
    start_fit_time = time.time()

    xgb_clf = XGBClassifier(eval_metric=['merror'],
                            objective='multi:softmax',
                            param_grid=best_params)

    clf = GridSearchCV(estimator=xgb_clf, param_grid=parameters, cv=10, n_jobs=-1)

    # fit model
    clf.fit(X_train, y_train)
    end_fit_time = time.time()

    # make predictions
    start_pred_time = time.time()

    y_pred = clf.predict(X_test)
    print(f"Accuracy on test data: {accuracy_score(y_test, y_pred) * 100:.3f}")
    end_pred_time = time.time()

    print(f"Time taken to run training code: {end_fit_time - start_fit_time} seconds")
    print(f"Time taken to run prediction code: {end_pred_time - start_pred_time} seconds")

    print(clf.best_params_)

    return clf, y_pred
