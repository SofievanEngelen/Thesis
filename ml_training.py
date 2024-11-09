from pathlib import Path
from collections import Counter
import pickle
from typing import Callable, Optional, Union
from sklearn.model_selection import train_test_split, GridSearchCV, LeaveOneGroupOut
from imblearn.over_sampling import SMOTE
from sklearn import linear_model, ensemble, svm, naive_bayes
import xgboost as xgb
import pandas as pd
from constants import *
from feature_selection import PCA_analysis


def load_or_train_model(file_path: Union[str, Path], train_func: Callable[[], object]) -> object:
    """
    Loads a model from a file if it exists, otherwise trains a new model.

    :param file_path: Path to the model file.
    :param train_func: A function that trains and returns a new model.
    :return: The loaded or newly trained model.
    """
    file_path = Path(file_path)

    if file_path.is_file():
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    return train_func()


def train_model(X: pd.DataFrame, y: pd.DataFrame, model: str, features: str, smote: bool,
                verbose: bool = True) -> object:
    """
    Trains a machine learning model with optional SMOTE resampling and feature selection.

    :param X: Input features as a DataFrame.
    :param y: Target labels as a DataFrame.
    :param model: Type of model to train. Options are ["XGBoost", "LogReg", "RandomForest", "LinearSVM", "NaiveBayes"].
    :param features: Feature set to use. Options are ["G", "L", "G+L"].
    :param smote: Whether to apply SMOTE for resampling the dataset.
    :param verbose: Whether to enable verbose logging. Default is True.
    :return: The trained model.
    """
    trained_model_path = f"pickles/{model}_trained_model_{features}_{'smote' if smote else 'no-smote'}.pkl"

    # Load or train the model
    clf = load_or_train_model(trained_model_path, lambda: _train_new_model(X, y, model, features, smote, verbose))

    return clf


def _train_new_model(X: pd.DataFrame, y: pd.DataFrame, model: str, features: str, smote: bool, verbose: bool = True) -> object:
    """
    Helper function to train a new model, with optional SMOTE resampling and feature selection.

    :param X: The input features as a DataFrame.
    :param y: The target labels as a DataFrame.
    :param model: The type of model to train.
    :param features: The feature set to use.
    :param smote: Whether to apply SMOTE for resampling.
    :param verbose: Whether to enable verbose logging.
    :return: The trained model.
    """
    log_message(f"Retrieving features for {features}...", verbose)

    # Feature selection based on the specified feature set
    feature_sets = {
        "G": GLOBAL_FEATURES,
        "L": LOCAL_FEATURES,
        "G+L": list(set(GLOBAL_FEATURES + LOCAL_FEATURES))
    }
    selected_features = feature_sets.get(features)
    X_features = X[selected_features]

    if not selected_features:
        raise ValueError(f"{features} is not a valid feature set.")

    log_message(f"Performing PCA analysis on selected features: {selected_features}...", verbose)

    X_pca_df, explained_variance_ratio, singular_values = PCA_analysis(X_features)

    log_message(f"Finished feature selection process, {len(X_pca_df.columns)} features left", verbose)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_pca_df, y, test_size=0.2)

    log_message(f"Features retrieved: {X_train.columns}", verbose)

    # Apply SMOTE for resampling if specified
    if smote:
        log_message("Applying SMOTE...", verbose)
        log_message(f'Original dataset shape: {Counter(y_train)}', verbose)
        sm = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=5, n_jobs=-1)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        log_message(f'Resampled dataset shape: {Counter(y_train)}', verbose)

    # Get model with hyperparameters
    clf = get_model_with_params(model, X_train, y_train)

    # Train the model
    log_message(f"Training model: {clf}...", verbose)
    clf.fit(X_train, y_train)
    log_message("Done training model.", verbose)

    # Save the trained model to a file
    trained_model_path = f"{model}_trained_model_{features}_{'smote' if smote else 'no-smote'}.pkl"
    with open(trained_model_path, 'wb') as f:
        pickle.dump(clf, f)

    log_message("Model saved.", verbose)
    return clf


def get_model_with_params(model_name: str, X_train: pd.DataFrame, y_train: pd.DataFrame) -> Union[
    xgb.XGBClassifier, linear_model.LogisticRegression, ensemble.RandomForestClassifier, svm.LinearSVC, naive_bayes.GaussianNB]:
    """
    Gets a model with the best hyperparameters using GridSearchCV and Leave-One-Group-Out cross-validation.

    :param model_name: The type of model to train. Options are ["XGBoost", "LogReg", "RandomForest", "LinearSVM", "NaiveBayes"].
    :param X_train: The training data as a DataFrame.
    :param y_train: The training labels as a DataFrame.
    :return: The model with the best hyperparameters.
    """
    # Mapping of model names to their classes and parameter grids
    model_mapping = {
        "XGBoost": (xgb.XGBClassifier, XGB_PARAMS),
        "LogReg": (linear_model.LogisticRegression, LR_PARAMS),
        "RandomForest": (ensemble.RandomForestClassifier, RF_PARAMS),
        "LinearSVM": (svm.LinearSVC, LSVM_PARAMS),
        "NaiveBayes": (naive_bayes.GaussianNB, None)  # No parameter tuning for NaiveBayes
    }

    model_info = model_mapping.get(model_name)
    if model_info is None:
        raise ValueError(f"{model_name} is not a valid model.")

    model_class, param_grid = model_info
    # If the model doesn't require parameter tuning, return it directly
    if param_grid is None:
        return model_class()

    # Perform hyperparameter tuning using Leave-One-Group-Out cross-validation
    groups = X_train["group_id"].values
    logo = LeaveOneGroupOut()
    gridsearch = GridSearchCV(model_class(), param_grid, cv=logo, scoring='f1', verbose=4, n_jobs=-1)
    gridsearch.fit(X_train, y_train, groups=groups)

    # Return the model with the best hyperparameters
    print(gridsearch.best_params_)
    return model_class(**gridsearch.best_params_)


# probes = pd.read_csv('Data/original-data/probe_data.csv')
# train_windows = pd.read_csv('Data/processed-data/train_windows_features.csv')
# print(train_model(X=train_windows, y=probes, model="XGBoost", features="G+L", smote=True))
