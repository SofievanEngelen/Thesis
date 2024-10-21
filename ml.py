import os
import pickle
from collections import Counter

import sklearn

from imblearn.over_sampling import SMOTE
import xgboost as xgb
from constants import *

print(xgb.XGBClassifier().get_params())
# {'objective': 'binary:logistic, 'colsample_bytree': None, 'device': None,  'eval_metric': None, 'feature_types': None, 'gamma': None, 'grow_policy': None,
#    'importance_type': None, 'interaction_constraints': None, 'learning_rate': None, 'max_bin': None,
#    'max_cat_threshold': None, 'max_cat_to_onehot': None, 'max_delta_step': None, 'max_depth': None,
#    'max_leaves': None, 'min_child_weight': None, 'missing': nan, 'monotone_constraints': None, 'multi_strategy': None,
#    'n_estimators': None, 'n_jobs': None, 'num_parallel_tree': None, 'random_state': None, 'reg_alpha': None,
#    'reg_lambda': None, 'sampling_method': None, 'scale_pos_weight': None, 'subsample': None, 'tree_method': None,
#    'validate_parameters': None, 'verbosity': None}
print(sklearn.linear_model.LogisticRegression().get_params())
# {'C': 1.0, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'l1_ratio': None,
#    'max_iter': 100, 'multi_class': 'auto', 'n_jobs': None, 'penalty': 'l2', 'random_state': None, 'solver': 'lbfgs',
#    'tol': 0.0001, 'verbose': 0, 'warm_start': False}
print(sklearn.ensemble.RandomForestClassifier().get_params())
# {'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': None, 'max_features':
#    'sqrt', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1,
#    'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': 100,
#    'n_jobs': None, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False}
print(sklearn.svm.LinearSVC().get_params())


# {'C': 1.0, 'class_weight': None, 'dual': 'warn', 'fit_intercept': True, 'intercept_scaling': 1,
#    'loss': 'squared_hinge', 'max_iter': 1000, 'multi_class': 'ovr', 'penalty': 'l2', 'random_state': None,
#    'tol': 0.0001}


def train_model(X, y, model: str, features: str, smote: bool = True, verbose: bool = True):

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

    if verbose:
        print(f"Retrieving features for {features}...")

    match features:
        case "G":
            X = X[X.columns.intersection(GLOBAL_FEATURES)]
        case "L":
            X = X[X.columns.intersection(LOCAL_FEATURES)]
        case "G+L":
            feature_list = GLOBAL_FEATURES + LOCAL_FEATURES
            X = X[X.columns.intersection(feature_list)]
        case _:
            raise ValueError(f"{features} is not a valid feature set.")

    if verbose:
        print(f"Features retrieved: {X.columns}")

    if smote:
        if verbose:
            print("Applying SMOTE...\n", f'Original dataset shape {Counter(y)}')

        sm = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=5, n_jobs=-1)
        X, y = sm.fit_resample(X, y)

        if verbose:
            print(f'Resampled dataset shape {Counter(y)}', "\nFinished SMOTE.")

    # match model:
    clf = get_model(model)

    if verbose:
        print(f"Training model: {clf}")

    clf.fit(X_train, y_train)

    if verbose:
        print(f"Done training model, testing model now.")

    clf.predict(X_test, y_test)


def get_model(model_name):
    model_path = f"{model_name}_model.pkl"

    if os.path.isfile(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)

    match model_name:
        case "XGBoost":
            return xgb.XGBClassifier(XGB_FINAL)
        case "LogReg":
            return sklearn.linear_model.LogisticRegression(LR_FINAL)
        case "RandomForest":
            return sklearn.ensemble.RandomForestClassifier(RF_FINAL)
        case "linearSVM":
            return sklearn.svm.LinearSVC(LSVM_PARAMS)
        case "NaiveBayes":
            return sklearn.naive_bayes.GaussianNB()
        case _:
            raise ValueError(f"{model_name} is not a valid model.")


def cross_validate_parameters(model, X_train, y_train):

    match model:
        case "XGBoost":
            model = xgb.XGBClassifier()
            param_grid = XGB_PARAMS
        case "LogReg":
            model = sklearn.linear_model.LogisticRegression()
            param_grid = LR_PARAMS
        case "RandomForest":
            model = sklearn.ensemble.RandomForestClassifier()
            param_grid = RF_PARAMS
        case "linearSVM":
            model = sklearn.svm.LinearSVC()
            param_grid = LSVM_PARAMS
        case "NaiveBayes":
            return sklearn.naive_bayes.GaussianNB()
        case _:
            raise ValueError(f"{model} is not a valid model.")

    clf = sklearn.model_selection.GridSearchCV(model, param_grid, scoring='f1', cv=5, verbose=4)
    clf.fit(X_train, y_train)
