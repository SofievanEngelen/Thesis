import numpy as np
import pandas as pd
from IPython.core.display_functions import display
from sklearn.dummy import DummyRegressor
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor


def train_model(data):
    X, y = data.loc[:, ~data.columns.isin(['MW-score-1', 'MW-score-2', 'MW-score-3', 'MW-score-4',
                                           'MW-score-5', 'MW-score-6', 'MW-score-7', 'Participant', 'index'])], \
        data.loc[:, data.columns.isin(
            ['MW-score-1', 'MW-score-2', 'MW-score-3', 'MW-score-4', 'MW-score-5', 'MW-score-6', 'MW-score-7'])]

    # One-hot encode 'Thought_Type'
    # y = OneHotEncoder().fit_transform(y_class.values.reshape(-1, 1)).toarray()

    # Impute missing values
    imputer = KNNImputer(n_neighbors=2, weights='uniform')
    X = imputer.fit_transform(X, y)

    train_p, test_p = train_test_split(data['Participant'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

    parameters = {'n_estimators': [60], 'max_depth': [1], 'learning_rate': [0.23]}

    # X, y = (data.loc[:, ~data.columns.isin(['MW-score-1', 'MW-score-2', 'MW-score-3', 'MW-score-4',
    #                                         'MW-score-5', 'MW-score-6', 'MW-score-7', 'index'])],
    #         data.loc[:, data.columns.isin(['Participant', 'MW-score-1', 'MW-score-2', 'MW-score-3', 'MW-score-4',
    #                                        'MW-score-5', 'MW-score-6', 'MW-score-7'])])
    # # Impute missing values
    # imputer = KNNImputer(n_neighbors=2, weights='uniform')
    # X_imputed = X.drop(['Participant'], axis=1)
    # X_imputed_array = imputer.fit_transform(X_imputed)
    # X_imputed = pd.DataFrame(X_imputed_array, columns=X_imputed.columns)
    # X_imputed['Participant'] = X['Participant']
    #
    # unique_participants = data['Participant'].unique()
    # train_p, test_p = train_test_split(unique_participants, test_size=0.2)
    # X_train, X_test, y_train, y_test = (X_imputed.loc[X_imputed['Participant'].isin(train_p)],
    #                                     X_imputed.loc[X_imputed['Participant'].isin(test_p)],
    #                                     y.loc[y['Participant'].isin(train_p)],
    #                                     y.loc[y['Participant'].isin(test_p)])
    #
    # X_train.drop(['Participant'], axis=1, inplace=True)
    # y_train.drop(['Participant'], axis=1, inplace=True)
    # X_test.drop(['Participant'], axis=1, inplace=True)
    # y_test.drop(['Participant'], axis=1, inplace=True)
    #
    # parameters = {'n_estimators': [108], 'max_depth': [2],
    #               'learning_rate': [0.01]}
    # n_estimators=108, max_depth=2, learning_rate=0.01
    # Create, fit, and evaluate XGBoost model
    xgb_reg = XGBRegressor(eval_metric=['rmse', 'mae', 'mape'],
                           objective='reg:squarederror'
                           )
    # n_estimators=108,
    # max_depth=2,
    # learning_rate=0.01)
    xgb_reg = GridSearchCV(estimator=xgb_reg, param_grid=parameters, cv=10, n_jobs=-1)
    xgb_reg.fit(X_train, y_train)
    y_xgb_pred = xgb_reg.predict(X_test)
    mse_xgb = mean_squared_error(y_test, y_xgb_pred)
    r2_xgb = r2_score(y_test, y_xgb_pred)
    print(f"Mean Squared Error XGB: {mse_xgb:.3f}")
    print(f"R-squared XGB: {r2_xgb:.3f}")

    # Create, fit, and evaluate Dummy model
    dum_reg = DummyRegressor(strategy="mean")
    dum_reg.fit(X_train, y_train)
    y_dum_pred = dum_reg.predict(X_test)
    mse_dum = mean_squared_error(y_test, y_dum_pred)
    r2_dum = r2_score(y_test, y_dum_pred)
    print(f"Mean Squared Error DummyRegressor: {mse_dum:.3f}")
    print(f"R-squared DummyRegressor: {r2_dum:.3f}")

    print(xgb_reg.best_params_)

    return xgb_reg, y_xgb_pred, y_dum_pred

    # Mean Squared Error: 0.114
    # {'learning_rate': 0.01, 'max_depth': 2, 'n_estimators': 500}


def evaluate_model(model, testdata):
    # Make predictions
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.3f}")
