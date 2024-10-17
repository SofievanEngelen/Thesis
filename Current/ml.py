import os
import pickle
from collections import Counter

import sklearn
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from constants import *


def train_model(X, y, model: str, features: str, smote: bool = True, verbose: bool = True):
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

    if smote:
        if verbose:
            print("Applying SMOTE...\n", f'Original dataset shape {Counter(y)}')

        sm = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=5, n_jobs=-1)
        X, y = sm.fit_resample(X, y)

        if verbose:
            print(f'Resampled dataset shape {Counter(y)}', "\nFinished SMOTE.")

    # match model:



def get_model(model_name):

    model_path = f"{model_name}_model.pkl"

    if os.path.isfile(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)

    match model_name:
        case "XGBoost":
            if os.path.isfile("xgboost_model.pkl"):
                # model =  # READ MODEL IN FROM PICKLE FILE

            pass
        case "LogReg":
            params = {'penalty': 'l2', 'C': 1.0, }
            sklearn.linear_model.LogisticRegression(params)
            pass
        case "RandomForest":
            pass
        case "linearSVM":
            pass
        case "NaiveBayes":
            pass
        case _:
            raise ValueError(f"{model_name} is not a valid model.")


def cross_validate_parameters(model):
    match model:
        case "XGBoost":
            clf = xgb.XGBClassifier(XGB_PARAMS)
        case "LogReg":
            clf = sklearn.linear_model.LogisticRegression(LR_PARAMS)
        case "RandomForest":
            clf = sklearn.ensemble.RandomForestClassifier(RF_PARAMS)
        case "linearSVM":
            clf = sklearn.svm.LinearSVC(LSVM_PARAMS)
        case "NaiveBayes":
            clf = sklearn.naive_bayes.GaussianNB(NB_PARAMS)
        case _:
            raise ValueError(f"{model} is not a valid model.")



