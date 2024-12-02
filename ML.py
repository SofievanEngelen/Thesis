from pathlib import Path
from collections import Counter
import pickle
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import (precision_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix,
                             ConfusionMatrixDisplay, balanced_accuracy_score, matthews_corrcoef)
from sklearn.model_selection import train_test_split, GridSearchCV, LeaveOneGroupOut
from imblearn.over_sampling import SMOTE
from sklearn import linear_model, ensemble, svm, naive_bayes
import xgboost as xgb
import pandas as pd
from constants import *
from feature_selection import PCA_analysis

# Mapping used to display SMOTE status
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

    # Get selected features for specified feature set
    selected_features = feature_map.get(feature_set)
    if not selected_features:
        raise ValueError(f"{feature_set} is not a valid feature set.")

    # Retrieve the selected features from the DataFrame
    X_features = X[selected_features]

    # Apply PCA analysis to the features
    X_pca_df = PCA_analysis(X_features, feature_set, verbose=False)

    # Optionally print the number of retained features after PCA
    if verbose:
        print(f"Applied PCA on {feature_set}: {len(X_pca_df.columns)} features retained.")
    return X_pca_df


def plot_confusion_matrix(actual: pd.Series, predicted: pd.Series, model: object, features: str,
                          use_smote: bool) -> None:
    """
    Plot and save confusion matrix for a given model and its predictions.

    :param actual: True labels.
    :param predicted: Predicted labels.
    :param model: Trained model.
    :param features: Feature subset used.
    :param use_smote: Whether SMOTE was used during training.
    """
    matrix = confusion_matrix(actual, predicted)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=[0, 1])
    cm_display.plot()
    plt.title(f"{type(model).__name__} ({features}, {'no' if not use_smote else ''} smote)")
    plt.savefig(f"Confusion_matrices/{type(model).__name__}_{features}_{smote_mapping[use_smote]}.png")


def save_metrics(metrics: Dict[str, float]) -> None:
    """
    Print model performance metrics.

    :param metrics: Dictionary of performance metrics.
    """
    for metric, score in metrics.items():
        print(f"{metric}: {score:.4f}")


def plot_feature_importances(model: str, clf: object, X_train: pd.DataFrame, model_id: str, feature_set: str,
                             use_smote: bool) -> None:
    """
    Calculate, save, and plot feature Feature Importances for a trained model.

    :param model: Model name.
    :param clf: Trained model.
    :param X_train: Training features DataFrame.
    :param model_id: Model identifier.
    :param feature_set: Feature subset used ('G', 'L', or 'G+L').
    :param use_smote: Whether SMOTE was used during training.
    """
    match model:
        case "LogReg":
            # Extract model coefficients and feature names
            coefficients = clf.coef_[0]
            feature_names = X_train.columns

            # Sort features by the absolute value of coefficients
            sorted_indices = np.argsort(np.abs(coefficients))[::-1]
            sorted_importances = np.abs(coefficients)[sorted_indices]
            sorted_feature_names = [feature_names[i] for i in sorted_indices]

            # Save feature Feature Importances to CSV
            pd.DataFrame({'Feature': sorted_feature_names, 'Importance': sorted_importances}).to_csv(
                f"Feature Importances/{model_id}_importances.csv", header=True, index=False)

            # Plot feature Feature Importances
            plt.figure(figsize=(10, 10))
            plt.barh(sorted_feature_names, sorted_importances, color='skyblue')
            plt.xlabel("Importance (|Coefficient|)")
            plt.ylabel("Features")
            plt.title("Feature Importances (Logistic Regression)")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()

        case "RandomForest":
            # Extract feature Feature Importances from RandomForest
            X_temp_train = X_train.drop(columns=["group_id"], errors='ignore')
            importances = clf.feature_importances_
            feature_names = X_temp_train.columns

            # Create and save feature Feature Importances to CSV
            importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            importance_df = importance_df.sort_values(by='Importance', ascending=False)
            importance_df.to_csv(f"Feature Importances/{model_id}_importances.csv", header=True, index=False)

            # Plot feature Feature Importances
            plt.figure(figsize=(10, 10))
            plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
            plt.xlabel("Feature Importance")
            plt.ylabel("Features")
            plt.title(f"Random Forest Feature Importances {feature_set}, {smote_mapping[use_smote]}")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()

        case "XGBoost":
            # Extract feature Feature Importances from XGBoost
            feature_importances = clf.get_booster().get_score(importance_type='gain')
            importance_df = pd.DataFrame(
                {'Feature': list(feature_importances.keys()), 'Importance': list(feature_importances.values())})
            importance_df = importance_df.sort_values(by='Importance', ascending=False)
            importance_df.to_csv(f"Feature Importances/{model_id}_importances.csv", header=True, index=False)

            # Plot feature Feature Importances
            plt.figure(figsize=(10, 10))
            plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
            plt.xlabel("Feature Importance (Gain)")
            plt.ylabel("Features")
            plt.title(f"XGBoost Feature Importances {feature_set}, {smote_mapping[use_smote]}")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()

        case "LinearSVM":
            # Extract feature Feature Importances from LinearSVM
            coefficients = clf.coef_[0]
            feature_names = X_train.columns

            # Sort features by the absolute value of coefficients
            sorted_indices = np.argsort(np.abs(coefficients))[::-1]
            sorted_importances = np.abs(coefficients)[sorted_indices]
            sorted_feature_names = [feature_names[i] for i in sorted_indices]

            # Save feature Feature Importances to CSV
            pd.DataFrame({'Feature': sorted_feature_names, 'Importance': sorted_importances}).to_csv(
                f"Feature Importances/{model_id}_importances.csv", header=True, index=False)

            # Plot feature Feature Importances
            plt.figure(figsize=(10, 10))
            plt.barh(sorted_feature_names, sorted_importances, color='skyblue')
            plt.xlabel("Importance (|Coefficient|)")
            plt.ylabel("Features")
            plt.title("Feature Importances (Linear SVM)")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()


def train_and_evaluate_models(model_list: List[str], X: pd.DataFrame, y: pd.Series, verbose: bool = True) -> Dict[
    str, Dict[str, float]]:
    """
    Train and evaluate a list of models on different feature sets, optionally with SMOTE.

    :param model_list: List of model names to train.
    :param X: Feature DataFrame.
    :param y: Target label Series.
    :param verbose: If True, print log messages.
    :return: Dictionary of models and their performance metrics.
    """
    metrics = {}

    for feature_set in ["G+L", "L", "G"]:
        # Retrieve PCA-transformed features for the specified feature set
        X_transformed = retrieve_features(X, feature_set, verbose)
        if "participant" in X_transformed.columns:
            X_transformed.drop(columns=["participant"], inplace=True)

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=RANDOM_SEED)

        # Train and evaluate each model
        for model in model_list:
            for use_smote in [True, False]:
                model_id = f"{model}_{feature_set}_{smote_mapping[use_smote]}"

                # Train the model
                clf = train_model(X_train, y_train, model, feature_set, use_smote, verbose)

                if "group_id" in X_test.columns:
                    X_test.drop(columns=["group_id"], inplace=True)

                # Make predictions
                y_pred = clf.predict(X_test)

                # Plot and save confusion matrix and feature Feature Importances
                plot_confusion_matrix(y_test, y_pred, clf, feature_set, use_smote)
                plot_feature_importances(model, clf, X_train, model_id, feature_set, use_smote)

                # Evaluate model and store metrics
                metrics[model_id] = evaluate_model(clf, y_pred, y_test, verbose)

    return metrics


def evaluate_model(clf: object, y_pred: pd.Series, y_test: pd.Series, verbose: bool = True) -> Dict[str, float]:
    """
    Evaluate a trained model using test data and calculate various performance metrics.

    :param clf: Trained model.
    :param y_pred: Predicted labels.
    :param y_test: True labels.
    :param verbose: If True, print log messages.
    :return: Dictionary of evaluation metrics.
    """
    metrics = {
        "Balanced Accuracy": round(balanced_accuracy_score(y_test, y_pred), 5),
        "Precision": round(precision_score(y_test, y_pred), 5),
        "Recall": round(recall_score(y_test, y_pred), 5),
        "F1 Score": round(f1_score(y_test, y_pred), 5),
        "Cohen's kappa": round(cohen_kappa_score(y_test, y_pred), 5),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 5)
    }

    if verbose:
        print(f"Evaluation Metrics for {type(clf).__name__}:")
        save_metrics(metrics)

    return metrics


def train_model(X_train: pd.DataFrame, y_train: pd.Series, model_name: str, feature_set: str, use_smote: bool,
                verbose: bool = True) -> object:
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
    model_path = Path(f"p/{model_name}_trained_model_{feature_set}_{smote_mapping[use_smote]}.pkl")
    return load_or_train_model(model_path,
                               lambda: _train_new_model(X_train, y_train, model_name, feature_set, use_smote, verbose))


def _train_new_model(X_train: pd.DataFrame, y_train: pd.Series, model_name: str, feature_set: str, use_smote: bool,
                     verbose: bool = True) -> object:
    """
    Train a new model with specified settings, applying SMOTE if required.

    :param X_train: Training features DataFrame.
    :param y_train: Training labels Series.
    :param model_name: Model name for training.
    :param use_smote: If True, apply SMOTE.
    :param verbose: If True, print log messages.
    :return: Trained model.
    """
    # Apply SMOTE if necessary
    if use_smote:
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        if verbose:
            print(f"Applied SMOTE: {Counter(y_train)}")

    # Drop "group_id" if it is in the training data
    X_temp_train = X_train.drop(columns=["group_id"], errors='ignore')

    # Train the model with optimal hyperparameters
    clf = get_model_with_params(model_name, feature_set, use_smote, X_train.get("group_id", len(X_train)), X_temp_train,
                                y_train, verbose)
    clf.fit(X_temp_train, y_train)

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

    # Train a new model and save it to file
    model = train_func()
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)
    return model


def get_model_with_params(model_name: str, feature_set: str, use_smote: bool, groups: pd.Series, X_train: pd.DataFrame,
                          y_train: pd.Series, verbose: bool = True) -> object:
    """
    Get model with optimal hyperparameters using GridSearchCV and Leave-One-Group-Out CV.

    :param model_name: Name of the model to instantiate.
    :param feature_set: Feature subset used ('G', 'L', or 'G+L').
    :param use_smote: Whether SMOTE was used.
    :param groups: Groups for Leave-One-Group-Out cross-validation.
    :param X_train: Training data.
    :param y_train: Training labels.
    :param verbose: Boolean indicating whether to log progress to console.
    :return: Model with best hyperparameters.
    """
    # Define model mapping with hyperparameter grids
    model_map = {
        "XGBoost": (xgb.XGBClassifier, XGB_PARAMS),
        "LogReg": (linear_model.LogisticRegression, LR_PARAMS),
        "RandomForest": (ensemble.RandomForestClassifier, RF_PARAMS),
        "LinearSVM": (svm.LinearSVC, LSVM_PARAMS),
        "NaiveBayes": (naive_bayes.GaussianNB, None)
    }

    # Retrieve model class and parameter grid
    model_class, param_grid = model_map.get(model_name, (None, None))

    # Handle XGBoost-specific adjustments for class imbalance
    if model_name == "XGBoost":
        num_pos = y_train.value_counts()[1]
        num_neg = y_train.value_counts()[0]
        param_grid['scale_pos_weight'] = [num_neg / num_pos]

    if not model_class:
        raise ValueError(f"{model_name} is not a valid model.")

    # If no hyperparameters are provided, instantiate the model directly
    if param_grid is None:
        return model_class()

    # Run GridSearchCV using Leave-One-Group-Out cross-validation
    logo = LeaveOneGroupOut()
    gridsearch = GridSearchCV(model_class(), param_grid, cv=logo, scoring='f1', n_jobs=-1)
    gridsearch.fit(X_train, y_train, groups=groups)

    if verbose:
        print(f"Found best parameters for {model_name}: {gridsearch.best_params_}")

    return model_class(**gridsearch.best_params_)
