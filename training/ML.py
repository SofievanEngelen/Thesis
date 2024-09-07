import pandas as pd
from IPython.core.display_functions import display
from sklearn.dummy import DummyRegressor
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, GroupShuffleSplit
from xgboost import XGBRegressor


def train_model(data):
    X, y = (data.loc[:, ~data.columns.isin(['MW-score-1', 'MW-score-2', 'MW-score-3', 'MW-score-4',
                                            'MW-score-5', 'MW-score-6', 'MW-score-7', 'index',
                                            'Probe'])],
            data.loc[:, data.columns.isin(
                ['Participant', 'MW-score-1', 'MW-score-2', 'MW-score-3', 'MW-score-4', 'MW-score-5', 'MW-score-6', 'MW-score-7'])])

    # # # Initialize GroupShuffleSplit
    # gss = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    # #
    # # # Perform the split
    # train_indices, test_indices = next(gss.split(data, groups=data['Participant']))
    # #
    # # Impute missing values
    # imputer = KNNImputer(n_neighbors=2, weights='uniform')
    # X = imputer.fit_transform(X, y)
    # X = pd.DataFrame(X)

    # print(X)
    #
    # # Split the data based on the indices
    # X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    # y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

    # #
    # train_p, test_p = train_test_split(data['Participant'])
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

    # print(X_train.head(), X_test.head())
    # X_train = X[297:]
    # X_test = X[:297]
    # y_train = y[297:]
    # y_test = y[:297]
    #
    # display(X)
    # display(y_train)
    #
    # pd.DataFrame(X_train).to_csv('D-split-tts-X_train.csv', index=False)
    # pd.DataFrame(X_test).to_csv('D-split-tts-X_test.csv', index=False)
    # pd.DataFrame(y_train).to_csv('D-split-tts-y_train.csv', index=False)
    # pd.DataFrame(y_test).to_csv('D-split-tts-y_test.csv', index=False)
    #
    # print(type(X_train), X_train.shape)
    # # display(X_train)
    # print(type(X_test), X_test.shape)
    # # display(X_test)
    # print(type(y_train), y_train.shape)
    # # display(y_train)
    # print(type(y_test), y_test.shape)
    # # display(y_test)

    # GOOD RESULTS
    # <class 'numpy.ndarray'> (1185, 94)
    # <class 'numpy.ndarray'> (297, 94)
    # <class 'pandas.core.frame.DataFrame'> (1185, 7)
    # <class 'pandas.core.frame.DataFrame'> (297, 7)
    # [[9.65000000e+02 2.48091232e+05 1.92062000e+05 ... 8.67955651e+00
    #   7.19090000e+04 1.48861900e+06]
    #  [9.47000000e+02 1.56098272e+05 1.44016500e+05 ... 4.31628892e+01
    #   9.40745000e+04 1.07986250e+07]
    #  [1.21450000e+03 1.88470543e+05 1.73999750e+05 ... 1.84348291e+01
    #   7.99210000e+04 7.12157000e+05]
    #  ...
    #  [7.92500000e+02 1.68126748e+05 1.51998250e+05 ... 1.20690190e+01
    #   7.58675000e+04 1.01379150e+06]
    #  [3.17000000e+02 1.16031989e+05 9.60485000e+04 ... 2.12860629e+01
    #   7.19850000e+04 1.66792900e+06]
    #  [6.80000000e+02 2.39533354e+05 1.99942500e+05 ... 1.26003210e+01
    #   8.18460000e+04 1.07110800e+06]]
    # <class 'numpy.ndarray'> (297, 94)
    # [[ 7.06500000e+02  2.26893961e+05  1.98949500e+05 ... -4.31555631e-01
    #    7.20055000e+04  2.75949000e+05]
    #  [ 7.90000000e+01  1.69675158e+05  1.06036500e+05 ...  9.71576229e-01
    #    7.19770000e+04  4.52004000e+05]
    #  [ 7.83500000e+02  1.61251563e+05  1.45942000e+05 ...  5.40031149e+00
    #    1.39920000e+05  4.59921000e+05]
    #  ...
    #  [ 7.71500000e+02  1.77160019e+05  1.55956000e+05 ...  2.96484054e+01
    #    8.00095000e+04  6.82685000e+06]
    #  [ 8.75500000e+02  1.98369733e+05  1.63935250e+05 ... -1.47058068e-01
    #    1.40022500e+05  2.43951000e+05]
    #  [ 6.28500000e+02  2.49599138e+05  2.21920250e+05 ...  2.10260379e+01
    #    7.79935000e+04  2.11637400e+06]]

    # BAD RESULTS
    # <class 'numpy.ndarray'> (1185, 94)
    # <class 'numpy.ndarray'> (297, 94)
    # <class 'pandas.core.frame.DataFrame'> (1185, 7)
    # <class 'pandas.core.frame.DataFrame'> (297, 7)
    # parameters = {'n_estimators': [60], 'max_depth': [1], 'learning_rate': [0.23]}

    #
    unique_participants = data['Participant'].unique()
    train_p, test_p = train_test_split(unique_participants, test_size=0.01, random_state=42)
    print(train_p, test_p)
    X_train, X_test, y_train, y_test = (X.loc[X['Participant'].isin(train_p)],
                                        X.loc[X['Participant'].isin(test_p)],
                                        y.loc[y['Participant'].isin(train_p)],
                                        y.loc[y['Participant'].isin(test_p)])

    display(X_train, X_test)
    #
    # train_data = data[data['Participant'].isin(train_p)]
    # test_data = data[data['Participant'].isin(test_p)]
    #
    # X_train, y_train = (
    #     train_data.loc[:, ~train_data.columns.isin(['MW-score-1', 'MW-score-2', 'MW-score-3', 'MW-score-4',
    #                                                 'MW-score-5', 'MW-score-6', 'MW-score-7', 'Participant', 'index',
    #                                                 'Probe'])],
    #     train_data.loc[:, train_data.columns.isin(['MW-score-1', 'MW-score-2', 'MW-score-3', 'MW-score-4',
    #                                                'MW-score-5', 'MW-score-6', 'MW-score-7'])])
    #
    # X_test, y_test = (test_data.loc[:, ~test_data.columns.isin(['MW-score-1', 'MW-score-2', 'MW-score-3', 'MW-score-4',
    #                                                             'MW-score-5', 'MW-score-6', 'MW-score-7', 'index',
    #                                                             'Participant', 'Probe'])],
    #                   test_data.loc[:, test_data.columns.isin(['MW-score-1', 'MW-score-2', 'MW-score-3', 'MW-score-4',
    #                                                            'MW-score-5', 'MW-score-6', 'MW-score-7'])])

    # X_train = X[:285]
    # X_test = X[285:]
    # y_train = y[:285]
    # y_test = y[285:]

    # display(X_test, y_test)

    X_train.drop(['Participant'], axis=1, inplace=True)
    y_train.drop(['Participant'], axis=1, inplace=True)
    X_test.drop(['Participant'], axis=1, inplace=True)
    y_test.drop(['Participant'], axis=1, inplace=True)

    # # Impute missing values
    imputer = KNNImputer(n_neighbors=2, weights='uniform')
    X_train = imputer.fit_transform(X_train, y_train)
    X_test = imputer.fit_transform(X_test, y_test)

    # print(X_train.shape)
    # print(X_test.shape)
    # print(y_train.shape)
    # print(y_test.shape)

    # pd.DataFrame(X_train).to_csv('P-split-X_train.csv', index=False)
    # pd.DataFrame(X_test).to_csv('P-split-X_test.csv', index=False)
    # pd.DataFrame(y_train).to_csv('P-split-y_train.csv', index=False)
    # pd.DataFrame(y_test).to_csv('P-split-y_test.csv', index=False)

    parameters = {'n_estimators': [108], 'max_depth': [2],
                  'learning_rate': [0.01]}

    # n_estimators=108, max_depth=2, learning_rate=0.01
    # Create, fit, and evaluate XGBoost model
    xgb_reg = XGBRegressor(eval_metric=['rmse', 'mae', 'mape'], objective='reg:squarederror')
    xgb_reg = GridSearchCV(estimator=xgb_reg, param_grid=parameters, cv=10, n_jobs=-1)
    xgb_reg.fit(X_train, y_train)
    y_xgb_pred = xgb_reg.predict(X_test)
    mse_xgb = mean_squared_error(y_test, y_xgb_pred)
    r2_xgb = r2_score(y_test, y_xgb_pred)
    print(f"Mean Squared Error XGB: {mse_xgb:.3f}")
    print(f"R-squared XGB: {r2_xgb:.3f}")

    linreg_reg = LinearRegression()
    linreg_reg.fit(X_train, y_train)
    y_linreg_pred = linreg_reg.predict(X_test)
    mse_linreg = mean_squared_error(y_test, y_linreg_pred)
    r2_linreg = r2_score(y_test, y_linreg_pred)
    print(f"Mean Squared Error Linear Regressor: {mse_linreg:.3f}")
    print(f"R-squared Linear Regressor: {r2_linreg:.3f}")

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
