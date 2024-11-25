from pathlib import Path
from collections import Counter
import pickle
from typing import Callable, List, Dict

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score, \
    confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score, matthews_corrcoef
from sklearn.model_selection import train_test_split, GridSearchCV, LeaveOneGroupOut
from imblearn.over_sampling import SMOTE
from sklearn import linear_model, ensemble, svm, naive_bayes
import xgboost as xgb
import pandas as pd

from constants import *
from feature_selection import PCA_analysis

smote_mapping = {True: "smote", False: "no_smote"}
param_mapping = {
    'XGBoost G+L True': {'objective': 'binary:logistic', 'base_score': None, 'booster': None, 'colsample_bylevel': None,
                         'colsample_bynode': None, 'colsample_bytree': 0.8, 'device': None, 'eval_metric': None,
                         'gamma': 0, 'grow_policy': None, 'interaction_constraints': None, 'learning_rate': None,
                         'max_bin': None, 'max_cat_threshold': None, 'max_cat_to_onehot': None, 'max_delta_step': None,
                         'max_depth': 7, 'max_leaves': None, 'min_child_weight': 1, 'monotone_constraints': None,
                         'multi_strategy': None, 'n_jobs': None, 'num_parallel_tree': None, 'random_state': None,
                         'reg_alpha': None, 'reg_lambda': None, 'sampling_method': None, 'scale_pos_weight': 1.0,
                         'subsample': 0.6, 'tree_method': 'hist', 'validate_parameters': None, 'verbosity': None,
                         'alpha': 0, 'eta': 0.05, 'lambda': 1},
    'XGBoost G+L False': {'objective': 'binary:logistic', 'base_score': None, 'booster': None,
                          'colsample_bylevel': None, 'colsample_bynode': None, 'colsample_bytree': 0.4, 'device': None,
                          'eval_metric': None, 'gamma': 0, 'grow_policy': None, 'interaction_constraints': None,
                          'learning_rate': None, 'max_bin': None, 'max_cat_threshold': None, 'max_cat_to_onehot': None,
                          'max_delta_step': None, 'max_depth': 7, 'max_leaves': None, 'min_child_weight': 10,
                          'monotone_constraints': None, 'multi_strategy': None, 'n_jobs': None,
                          'num_parallel_tree': None, 'random_state': None, 'reg_alpha': None, 'reg_lambda': None,
                          'sampling_method': None, 'scale_pos_weight': 1.88, 'subsample': 0.6, 'tree_method': 'hist',
                          'validate_parameters': None, 'verbosity': None, 'alpha': 0, 'eta': 0.1, 'lambda': 1},
    'LogReg G+L True': {'C': 0.01, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1,
                        'l1_ratio': None, 'max_iter': 5000, 'multi_class': 'auto', 'n_jobs': None, 'penalty': 'l2',
                        'random_state': None, 'solver': 'lbfgs', 'tol': 0.0001, 'verbose': 0, 'warm_start': False},
    'LogReg G+L False': {'C': 0.01, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1,
                         'l1_ratio': None, 'max_iter': 500, 'multi_class': 'auto', 'n_jobs': None, 'penalty': 'l2',
                         'random_state': None, 'solver': 'lbfgs', 'tol': 0.0001, 'verbose': 0, 'warm_start': False},
    'RandomForest G+L True': {'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini',
                              'max_depth': 20, 'max_features': 10, 'max_leaf_nodes': None, 'max_samples': None,
                              'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2,
                              'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': 200,
                              'n_jobs': None, 'oob_score': False, 'random_state': None, 'verbose': 0,
                              'warm_start': False},
    'RandomForest G+L False': {'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini',
                               'max_depth': 20, 'max_features': 10, 'max_leaf_nodes': None, 'max_samples': None,
                               'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2,
                               'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': 100,
                               'n_jobs': None, 'oob_score': False, 'random_state': None, 'verbose': 0,
                               'warm_start': False},
    'LinearSVM G+L True': {'C': 0.001, 'class_weight': None, 'dual': 'warn', 'fit_intercept': True,
                           'intercept_scaling': 1, 'loss': 'squared_hinge', 'max_iter': 1000, 'multi_class': 'ovr',
                           'penalty': 'l2', 'random_state': None, 'tol': 0.0001, 'verbose': 0},
    'LinearSVM G+L False': {'C': 0.1, 'class_weight': None, 'dual': 'warn', 'fit_intercept': True,
                            'intercept_scaling': 1, 'loss': 'squared_hinge', 'max_iter': 1000, 'multi_class': 'ovr',
                            'penalty': 'l2', 'random_state': None, 'tol': 0.0001, 'verbose': 0},
    'NaiveBayes G+L True': {'priors': None, 'var_smoothing': 1e-09},
    'NaiveBayes G+L False': {'priors': None, 'var_smoothing': 1e-09},
    'XGBoost G True': {'objective': 'binary:logistic', 'base_score': None, 'booster': None, 'colsample_bylevel': None,
                       'colsample_bynode': None, 'colsample_bytree': 0.8, 'device': None, 'eval_metric': None,
                       'gamma': 0, 'grow_policy': None, 'interaction_constraints': None, 'learning_rate': None,
                       'max_bin': None, 'max_cat_threshold': None, 'max_cat_to_onehot': None, 'max_delta_step': None,
                       'max_depth': 7, 'max_leaves': None, 'min_child_weight': 10, 'monotone_constraints': None,
                       'multi_strategy': None, 'n_jobs': None, 'num_parallel_tree': None, 'random_state': None,
                       'reg_alpha': None, 'reg_lambda': None, 'sampling_method': None, 'scale_pos_weight': 1.0,
                       'subsample': 0.6, 'tree_method': 'hist', 'validate_parameters': None, 'verbosity': None,
                       'alpha': 0, 'eta': 0.1, 'lambda': 1},
    'XGBoost G False': {'objective': 'binary:logistic', 'base_score': None, 'booster': None, 'colsample_bylevel': None,
                        'colsample_bynode': None, 'colsample_bytree': 0.4, 'device': None, 'eval_metric': None,
                        'gamma': 0, 'grow_policy': None, 'interaction_constraints': None, 'learning_rate': None,
                        'max_bin': None, 'max_cat_threshold': None, 'max_cat_to_onehot': None, 'max_delta_step': None,
                        'max_depth': 3, 'max_leaves': None, 'min_child_weight': 5, 'monotone_constraints': None,
                        'multi_strategy': None, 'n_jobs': None, 'num_parallel_tree': None, 'random_state': None,
                        'reg_alpha': None, 'reg_lambda': None, 'sampling_method': None, 'scale_pos_weight': 1.88,
                        'subsample': 0.7, 'tree_method': 'hist', 'validate_parameters': None, 'verbosity': None,
                        'alpha': 0, 'eta': 0.1, 'lambda': 1},
    'LogReg G True': {'C': 0.01, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1,
                      'l1_ratio': None, 'max_iter': 500, 'multi_class': 'auto', 'n_jobs': None, 'penalty': 'l2',
                      'random_state': None, 'solver': 'liblinear', 'tol': 0.0001, 'verbose': 0, 'warm_start': False},
    'LogReg G False': {'C': 1, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1,
                       'l1_ratio': None, 'max_iter': 750, 'multi_class': 'auto', 'n_jobs': None, 'penalty': 'l2',
                       'random_state': None, 'solver': 'lbfgs', 'tol': 0.0001, 'verbose': 0, 'warm_start': False},
    'RandomForest G True': {'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini',
                            'max_depth': 20, 'max_features': 'log2', 'max_leaf_nodes': None, 'max_samples': None,
                            'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2,
                            'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': 500, 'n_jobs': None,
                            'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False},
    'RandomForest G False': {'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini',
                             'max_depth': 30, 'max_features': 10, 'max_leaf_nodes': None, 'max_samples': None,
                             'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 10,
                             'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': 100,
                             'n_jobs': None, 'oob_score': False, 'random_state': None, 'verbose': 0,
                             'warm_start': False},
    'LinearSVM G True': {'C': 0.001, 'class_weight': None, 'dual': 'warn', 'fit_intercept': True,
                         'intercept_scaling': 1, 'loss': 'squared_hinge', 'max_iter': 1000, 'multi_class': 'ovr',
                         'penalty': 'l2', 'random_state': None, 'tol': 0.0001, 'verbose': 0},
    'LinearSVM G False': {'C': 0.1, 'class_weight': None, 'dual': 'warn', 'fit_intercept': True, 'intercept_scaling': 1,
                          'loss': 'squared_hinge', 'max_iter': 1000, 'multi_class': 'ovr', 'penalty': 'l2',
                          'random_state': None, 'tol': 0.0001, 'verbose': 0},
    'NaiveBayes G True': {'priors': None, 'var_smoothing': 1e-09},
    'NaiveBayes G False': {'priors': None, 'var_smoothing': 1e-09},
    'XGBoost L True': {'objective': 'binary:logistic', 'base_score': None, 'booster': None, 'colsample_bylevel': None,
                       'colsample_bynode': None, 'colsample_bytree': 0.8, 'device': None, 'eval_metric': None,
                       'gamma': 0, 'grow_policy': None, 'interaction_constraints': None, 'learning_rate': None,
                       'max_bin': None, 'max_cat_threshold': None, 'max_cat_to_onehot': None, 'max_delta_step': None,
                       'max_depth': 7, 'max_leaves': None, 'min_child_weight': 1, 'monotone_constraints': None,
                       'multi_strategy': None, 'n_jobs': None, 'num_parallel_tree': None, 'random_state': None,
                       'reg_alpha': None, 'reg_lambda': None, 'sampling_method': None, 'scale_pos_weight': 1.0,
                       'subsample': 0.6, 'tree_method': 'hist', 'validate_parameters': None, 'verbosity': None,
                       'alpha': 0, 'eta': 0.05, 'lambda': 1},
    'XGBoost L False': {'objective': 'binary:logistic', 'base_score': None, 'booster': None, 'colsample_bylevel': None,
                        'colsample_bynode': None, 'colsample_bytree': 0.4, 'device': None, 'eval_metric': None,
                        'gamma': 0, 'grow_policy': None, 'interaction_constraints': None, 'learning_rate': None,
                        'max_bin': None, 'max_cat_threshold': None, 'max_cat_to_onehot': None, 'max_delta_step': None,
                        'max_depth': 3, 'max_leaves': None, 'min_child_weight': 10, 'monotone_constraints': None,
                        'multi_strategy': None, 'n_jobs': None, 'num_parallel_tree': None, 'random_state': None,
                        'reg_alpha': None, 'reg_lambda': None, 'sampling_method': None, 'scale_pos_weight': 1.88,
                        'subsample': 0.7, 'tree_method': 'hist', 'validate_parameters': None, 'verbosity': None,
                        'alpha': 0, 'eta': 0.1, 'lambda': 1},
    'LogReg L True': {'C': 0.01, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1,
                      'l1_ratio': None, 'max_iter': 500, 'multi_class': 'auto', 'n_jobs': None, 'penalty': 'l2',
                      'random_state': None, 'solver': 'liblinear', 'tol': 0.0001, 'verbose': 0, 'warm_start': False},
    'LogReg L False': {'C': 0.0001, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1,
                       'l1_ratio': None, 'max_iter': 500, 'multi_class': 'auto', 'n_jobs': None, 'penalty': 'l2',
                       'random_state': None, 'solver': 'liblinear', 'tol': 0.0001, 'verbose': 0, 'warm_start': False},
    'RandomForest L True': {'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini',
                            'max_depth': 30, 'max_features': 'log2', 'max_leaf_nodes': None, 'max_samples': None,
                            'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2,
                            'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': 500, 'n_jobs': None,
                            'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False},
    'RandomForest L False': {'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini',
                             'max_depth': 30, 'max_features': 10, 'max_leaf_nodes': None, 'max_samples': None,
                             'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 5,
                             'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': 200,
                             'n_jobs': None, 'oob_score': False, 'random_state': None, 'verbose': 0,
                             'warm_start': False},
    'LinearSVM L True': {'C': 1, 'class_weight': None, 'dual': 'warn', 'fit_intercept': True, 'intercept_scaling': 1,
                         'loss': 'squared_hinge', 'max_iter': 1000, 'multi_class': 'ovr', 'penalty': 'l2',
                         'random_state': None, 'tol': 0.0001, 'verbose': 0},
    'LinearSVM L False': {'C': 0.001, 'class_weight': None, 'dual': 'warn', 'fit_intercept': True,
                          'intercept_scaling': 1, 'loss': 'squared_hinge', 'max_iter': 1000, 'multi_class': 'ovr',
                          'penalty': 'l2', 'random_state': None, 'tol': 0.0001, 'verbose': 0},
    'NaiveBayes L True': {'priors': None, 'var_smoothing': 1e-09},
    'NaiveBayes L False': {'priors': None, 'var_smoothing': 1e-09}}


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
    X_pca_df = PCA_analysis(X_features, feature_set, verbose=False)
    if verbose:
        print(f"Applied PCA on {feature_set}: {len(X_pca_df.columns)} features retained.")
    return X_pca_df


def plot_confusion_matrix(actual, predicted, model, features, use_smote):
    matrix = confusion_matrix(actual, predicted)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=[0, 1])
    cm_display.plot()
    plt.title(f"{type(model).__name__} ({features}, {'no' if not use_smote else ''} smote)")
    plt.show()


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

    for feature_set in ["G+L", "G", "L"]:
        X_transformed = retrieve_features(X, feature_set, verbose)
        X_transformed.drop(columns=["Participant"], inplace=True)

        X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=RANDOM_SEED)

        for model in model_list:
            for use_smote in [True, False]:
                global model_id
                model_id = f"{model}_{feature_set}_{smote_mapping[use_smote]}"

                clf = train_model(X_train, y_train, model, feature_set, use_smote, verbose)
                print(model, feature_set, use_smote)

                if "group_id" in X_test.columns:
                    X_test.drop(columns=["group_id"], inplace=True)
                    # X_train.drop(columns=["group_id"], inplace=True)

                if model == "LogReg":
                    coefficients = clf.coef_[0]
                    # Example feature names for visualization (replace with actual feature names if available)
                    feature_names = [f"Feature {i}" for i in range(len(coefficients))]

                    # Sort feature importance and names for visualization
                    sorted_indices = np.argsort(np.abs(coefficients))[::-1]
                    sorted_importances = np.abs(coefficients)[sorted_indices]
                    sorted_feature_names = [feature_names[i] for i in sorted_indices]

                    # Plot the feature importance
                    plt.figure(figsize=(10, 6))
                    plt.barh(sorted_feature_names, sorted_importances, color='skyblue')
                    plt.xlabel("Importance (|Coefficient|)")
                    plt.ylabel("Features")
                    plt.title("Feature Importances (Logistic Regression)")
                    plt.gca().invert_yaxis()  # Invert y-axis to show the most important features at the top
                    plt.tight_layout()
                    plt.show()
                if model == "RandomForest":
                    X_temp_train = X_train.drop(columns=["group_id"])
                    importances = clf.feature_importances_
                    feature_names = X_temp_train.columns
                    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                    importance_df = importance_df.sort_values(by='Importance', ascending=False)
                    plt.figure(figsize=(10, 6))
                    plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
                    plt.xlabel('Feature Importance')
                    plt.ylabel('Features')
                    plt.title(f'Random Forest Feature Importances {feature_set}, {smote_mapping[use_smote]}')
                    plt.gca().invert_yaxis()  # Invert y-axis for better readability
                    plt.show()
                elif model == "XGBoost":
                    feature_importances = clf.get_booster().get_score(
                        importance_type='gain')  # Change to 'weight' or 'cover' if needed
                    importance_df = pd.DataFrame({'Feature': list(feature_importances.keys()),
                                                  'Importance': list(feature_importances.values())})
                    importance_df = importance_df.sort_values(by='Importance', ascending=False)
                    plt.figure(figsize=(10, 6))
                    plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
                    plt.xlabel('Feature Importance')
                    plt.ylabel('Features')
                    plt.title(f'XGBoost Feature Importances (Gain) {feature_set}, {smote_mapping[use_smote]}')
                    plt.gca().invert_yaxis()  # Invert y-axis for better readability
                    plt.show()

                y_pred = clf.predict(X_test)
                # plot_confusion_matrix(y_test, y_pred, clf, feature_set, use_smote)
                metrics[model_id] = evaluate_model(clf, y_pred, y_test, verbose)
    return metrics


def evaluate_model(clf: object, y_pred: pd.DataFrame, y_test: pd.Series, verbose: bool = True) -> Dict[str, float]:
    """
    Evaluate a trained model using test data and calculate various performance metrics.

    :param clf: Trained model.
    :param y_pred: Test feature DataFrame.
    :param y_test: Test label Series.
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
        if isinstance(clf, xgb.sklearn.XGBClassifier):
            print(clf.get_xgb_params())
        else:
            print(clf.get_params())
        # print(f"Evaluation Metrics for {clf}:")
        # save_metrics(metrics)

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
    model_path = Path(f"p/{model_name}_trained_model_{feature_set}_{smote_mapping[use_smote]}.pkl")
    return load_or_train_model(model_path,
                               lambda: _train_new_model(X_train, y_train, model_name, feature_set, use_smote, verbose))


def _train_new_model(
        X_train: pd.DataFrame, y_train: pd.Series, model_name: str, feature_set: str, use_smote: bool,
        verbose: bool = True
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

    groups = X_train.get("group_id", len(X_train))

    if "group_id" in X_train.columns:
        X_temp_train = X_train.drop(columns=["group_id"])
    else:
        X_temp_train = X_train

    clf = get_model_with_params(model_name, feature_set, use_smote, groups, X_temp_train, y_train)

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

    model = train_func()
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)
    return model


def get_model_with_params(model_name: str, features, use_smote, groups, X_train: pd.DataFrame, y_train: pd.Series,
                          verbose: bool = True) -> object:
    """
    Get model with optimal hyperparameters using GridSearchCV and Leave-One-Group-Out CV.

    :param model_name: Name of the model to instantiate.
    :param X_train: Training data.
    :param y_train: Training labels.
    :param verbose: Boolean indicating whether to log progess to console.

    :return: Model with best hyperparameters.
    """
    model_map = {
        "XGBoost": (xgb.XGBClassifier, XGB_PARAMS),
        "LogReg": (linear_model.LogisticRegression, LR_PARAMS),
        "RandomForest": (ensemble.RandomForestClassifier, RF_PARAMS),
        "LinearSVM": (svm.LinearSVC, LSVM_PARAMS),
        "NaiveBayes": (naive_bayes.GaussianNB, None)
    }

    model_class, param_grid = model_map.get(model_name, (None, None))

    # if model_name == "XGBoost":
    #     num_pos = y_train.value_counts()[1]
    #     num_neg = y_train.value_counts()[0]
    #     param_grid['scale_pos_weight'] = [num_neg / num_pos]
    #
    if not model_class:
        raise ValueError(f"{model_name} is not a valid model.")

    if param_grid is None:
        return model_class()

    print(X_train.columns)
    logo = LeaveOneGroupOut()
    gridsearch = GridSearchCV(model_class(), param_grid, cv=logo, scoring='f1', n_jobs=-1)
    gridsearch.fit(X_train, y_train, groups=groups)

    log_message(f"Found best parameters for {model_name}: {gridsearch.best_params_}", verbose)
    # params = param_mapping[f"{model_name} {features} {use_smote}"]
    # print(params)
    return model_class(**gridsearch.best_params_)
