from pathlib import Path
from collections import Counter
import pickle
from typing import Callable, List, Dict

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, LeaveOneGroupOut, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn import linear_model, ensemble, svm, naive_bayes
import xgboost as xgb
import pandas as pd

from constants import *
from feature_selection import PCA_analysis

import warnings
warnings.filterwarnings("ignore")


smote_mapping = {True: "smote", False: "no_smote"}


def retrieve_features(X: pd.DataFrame, feature_set: str, verbose: bool = True) -> pd.DataFrame:
    """
    Retrieve and apply PCA to a specified feature set.

    :param X: Input DataFrame of features.
    :param feature_set: Feature subset to use ('G', 'L', or 'G+L').
    :param verbose: If True, print log messages.

    :return: DataFrame of PCA-transformed features.
    """
    feature_map = {
        "G": GLOBAL_FEATURES,
        "L": LOCAL_FEATURES,
        "G+L": list(set(GLOBAL_FEATURES + LOCAL_FEATURES))
    }
    selected_features = feature_map.get(feature_set)
    if not selected_features:
        raise ValueError(f"{feature_set} is not a valid feature set.")

    X_features = X[selected_features]
    X_pca_df, _, _ = PCA_analysis(X_features, verbose=False)
    if verbose:
        print(f"Applied PCA on {feature_set}: {len(X_pca_df.columns)} features retained.")
    return X_pca_df


def save_metrics(metrics: Dict[str, float]) -> None:
    """
    Print model performance metrics.

    :param metrics: Dictionary of performance metrics.
    """
    for metric, score in metrics.items():
        print(f"{metric}: {score:.4f}")


def train_and_evaluate_models(
        model_list: List[str], X: pd.DataFrame, y: pd.Series, verbose: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Train and evaluate a list of models on different feature sets, optionally with SMOTE.

    :param model_list: List of model names to train.
    :param X: Feature DataFrame.
    :param y: Target label Series.
    :param verbose: If True, print log messages.

    :return: Dictionary of models and their performance metrics.
    """
    metrics = {}

    for model in model_list:
        for feature_set in ["G+L", "G", "L"]:
            X_transformed = retrieve_features(X, feature_set, verbose)
            X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

            for use_smote in [True, False]:
                global model_id
                model_id = f"{model}_{feature_set}_{smote_mapping[use_smote]}"

                clf = train_model(X_train, y_train, model, feature_set, use_smote, verbose)
                metrics[model_id] = evaluate_model(clf, X_test, y_test, X_transformed, y, verbose)
    return metrics


def evaluate_model(
        clf: object, X_test: pd.DataFrame, y_test: pd.Series, X: pd.DataFrame, y: pd.Series, verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate a trained model using test data and calculate various performance metrics.

    :param clf: Trained model.
    :param X_test: Test feature DataFrame.
    :param y_test: Test label Series.
    :param X: Full feature DataFrame for cross-validation.
    :param y: Full label Series for cross-validation.
    :param verbose: If True, print log messages.

    :return: Dictionary of evaluation metrics.
    """
    y_pred = clf.predict(X_test)
    y_pred_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_pred_prob) if y_pred_prob is not None else np.nan
    }

    cv_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    metrics["CV Accuracy"] = cv_scores.mean()
    metrics["CV Std Dev"] = cv_scores.std()

    if verbose:
        print(f"Evaluation Metrics for {clf}:")
        save_metrics(metrics)

    return metrics


def train_model(
        X_train: pd.DataFrame, y_train: pd.Series, model_name: str, feature_set: str, use_smote: bool,
        verbose: bool = True
) -> object:
    """
    Train a specified model with optional SMOTE and save the model.

    :param X_train: Training features DataFrame.
    :param y_train: Training labels Series.
    :param model_name: Name of the model to train.
    :param feature_set: Feature subset used ('G', 'L', or 'G+L').
    :param use_smote: If True, apply SMOTE.
    :param verbose: If True, print log messages.

    :return: Trained model.
    """
    model_path = Path(f"pickles/{model_name}_trained_model_{feature_set}_{smote_mapping[use_smote]}.pkl")
    return load_or_train_model(model_path, lambda: _train_new_model(X_train, y_train, model_name, use_smote, verbose))


def _train_new_model(
        X_train: pd.DataFrame, y_train: pd.Series, model_name: str, use_smote: bool, verbose: bool = True
) -> object:
    """
    Train a new model with specified settings, applying SMOTE if required.

    :param X_train: Training features DataFrame.
    :param y_train: Training labels Series.
    :param model_name: Model name for training.
    :param use_smote: If True, apply SMOTE.
    :param verbose: If True, print log messages.

    :return: Trained model.
    """
    if use_smote:
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        if verbose:
            print(f"Applied SMOTE: {Counter(y_train)}")

    clf = get_model_with_params(model_name, X_train, y_train)
    clf.fit(X_train, y_train)
    if verbose:
        print(f"Trained {model_name} model with {smote_mapping[use_smote]}.")

    return clf


def load_or_train_model(file_path: Path, train_func: Callable[[], object]) -> object:
    """
    Load model from file or train a new model if file does not exist.

    :param file_path: Path to the model file.
    :param train_func: Function that trains a new model if needed.

    :return: Loaded or newly trained model.
    """
    if file_path.is_file():
        with open(file_path, 'rb') as file:
            return pickle.load(file)

    model = train_func()
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)
    return model


def get_model_with_params(model_name: str, X_train: pd.DataFrame, y_train: pd.Series, verbose: bool = True) -> object:
    """
    Get model with optimal hyperparameters using GridSearchCV and Leave-One-Group-Out CV.

    :param model_name: Name of the model to instantiate.
    :param X_train: Training data.
    :param y_train: Training labels.
    :param verbose: Boolean indicating whether to log progess to console.

    :return: Model with best hyperparameters.
    """
    model_map = {
        "XGBoost": (xgb.XGBClassifier, {'alpha': [0], 'colsample_bytree': [0.4], 'eta': [0.1], 'gamma': [0], 'lambda': [1], 'max_depth': [3], 'min_child_weight': [10], 'n_estimators': [1000], 'objective': ['binary:logistic'], 'subsample': [0.7], 'tree_method': ['hist']}),
        "LogReg": (linear_model.LogisticRegression, LR_PARAMS),
        "RandomForest": (ensemble.RandomForestClassifier, RF_PARAMS),
        "LinearSVM": (svm.LinearSVC, LSVM_PARAMS),
        "NaiveBayes": (naive_bayes.GaussianNB, None)
    }

    model_class, param_grid = model_map.get(model_name, (None, None))

    if model_name == "XGBoost":
        num_pos = y_train.value_counts()[1]
        num_neg = y_train.value_counts()[0]
        param_grid['scale_pos_weight'] = [num_neg / num_pos]

    if not model_class:
        raise ValueError(f"{model_name} is not a valid model.")

    if param_grid is None:
        return model_class()

    groups = X_train.get("group_id", np.arange(len(X_train)))  # Default groups if column missing
    logo = LeaveOneGroupOut()
    gridsearch = GridSearchCV(model_class(), param_grid, cv=logo, scoring='f1', n_jobs=-1, verbose=4)
    gridsearch.fit(X_train, y_train, groups=groups)

    log_message(f"Found best parameters for {model_name}: {gridsearch.best_params_}", verbose)
    return model_class(**gridsearch.best_params_)
