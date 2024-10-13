from collections import Counter

from imblearn.over_sampling import SMOTE
import constants


def train_model(X, y, model: str, features: str, smote: bool = True, verbose: bool = True):
    match features:
        case "G":
            X = X[X.columns.intersection(constants.GLOBAL_FEATURES)]
        case "L":
            X = X[X.columns.intersection(constants.LOCAL_FEATURES)]
        case "G+L":
            feature_list = constants.GLOBAL_FEATURES + constants.LOCAL_FEATURES
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

    match model:
        case "XGBoost":
            param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
            
            pass
        case "LogReg":
            pass
        case "RandomForest":
            pass
        case "linearSVM":
            pass
        case "NaiveBayes":
            pass
        case _:
            raise ValueError(f"{model} is not a valid model.")
