import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import normalize


def ml(data):

    X, y_class = data.loc[:, ~data.columns.isin(['Thought_Type', 'Participant', 'index', 'Unnamed: 0'])], data[
        'Thought_Type']

    # One-hot encode 'Thought_Type'
    y = OneHotEncoder().fit_transform(y_class.values.reshape(-1, 1)).toarray()

    # Impute missing values
    imputer = KNNImputer(n_neighbors=2, weights='uniform')
    X = imputer.fit_transform(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

    parameters = {'n_estimators': 60, 'max_depth': 1, 'learning_rate': 0.23}

    # Create model instance
    xgb_reg = XGBRegressor(eval_metric=['rmse', 'mae', 'mape'],
                           objective='reg:squarederror',
                           n_estimators=60,
                           max_depth=1,
                           learning_rate=0.23)

    # reg = GridSearchCV(estimator=xgb_reg, param_grid=parameters, cv=10, n_jobs=-1)

    # Fit model
    xgb_reg.fit(X_train, y_train)

    # Make predictions
    y_pred = xgb_reg.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.3f}")

    # print(xgb_reg.best_params_)

    return xgb_reg, y_pred

    # Mean Squared Error: 0.114
    # {'learning_rate': 0.01, 'max_depth': 2, 'n_estimators': 500}
