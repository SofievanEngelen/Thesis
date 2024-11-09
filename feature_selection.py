## Feature selection
# First PCA
# Then exhaustive selection to find best subset of components
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

from constants import log_message

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import numpy as np

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE


def log_message(message, verbose=True):
    if verbose:
        print(message)


def PCA_analysis(X, verbose=True):
    """
    Perform PCA on the data with MLE to determine the optimal number of components automatically.

    Parameters:
    - X: DataFrame of input features.
    - verbose: Boolean for verbosity control.

    Returns:
    - X_pca_df: DataFrame of PCA-transformed components.
    - explained_variance_ratio_: Array of variance explained by each component.
    - singular_values_: Array of singular values.
    """
    # Drop non-feature columns (assumes names; adjust as necessary)
    feature_cols = X.drop(columns=['participant', 'paragraph', 'trial', 'group_id'], errors='ignore')
    print(X.columns)

    # Initialize PCA with MLE for automatic component selection
    pca = PCA(n_components='mle')
    X_pca = pca.fit_transform(feature_cols)

    # Create DataFrame for PCA-transformed data
    X_pca_df = pd.DataFrame(X_pca, columns=[f'PC{i + 1}' for i in range(pca.n_components_)])

    # Log explained variance and singular values
    log_message(f"Explained Variance Ratio (first 5 components): {pca.explained_variance_ratio_[:5]}", verbose)
    log_message(f"Singular Values (first 5 components): {pca.singular_values_[:5]}", verbose)
    log_message(f"Transformed PCA Data Shape: {X_pca_df.shape}", verbose)

    # Save component loadings for interpretation
    loadings = pd.DataFrame(pca.components_, columns=feature_cols.columns, index=X_pca_df.columns)
    loadings.to_csv("pca_loadings.csv", index=True)

    print(X['participant'])
    X_pca_df['Participant'] = X['participant']
    X_pca_df['group_id'] = X['group_id']

    return X_pca_df, pca.explained_variance_ratio_, pca.singular_values_


def RFE_selection(X_pca_df, y, n_features_rfe=30, verbose=True):
    """
    Perform RFE on PCA-transformed data to select the most predictive components.

    Parameters:
    - X_pca_df: DataFrame of PCA-transformed features.
    - y: Target variable.
    - n_features_rfe: Number of components to select in RFE.
    - verbose: Boolean for verbosity control.

    Returns:
    - X_rfe_selected: DataFrame of selected components after RFE.
    """
    # Initialize classifier and RFE
    estimator = RandomForestClassifier(random_state=42)
    rfe = RFE(estimator, n_features_to_select=n_features_rfe)
    rfe.fit(X_pca_df, y)

    # Retrieve and log selected components
    selected_pca_components = X_pca_df.columns[rfe.support_]
    log_message(f"Selected PCA components by RFE: {selected_pca_components.tolist()}", verbose)

    # Filter the dataset to include only RFE-selected components
    X_rfe_selected = X_pca_df[selected_pca_components]

    return X_rfe_selected


from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def optimal_features_RFE(X_pca_df, y, max_features=30, verbose=True):
    """
    Perform RFE on PCA-transformed data and use cross-validation to find the optimal number of components.

    Parameters:
    - X_pca_df: DataFrame of PCA-transformed features.
    - y: Target variable.
    - max_features: Maximum number of features to test.
    - verbose: Boolean for verbosity control.

    Returns:
    - optimal_n_features: The optimal number of features based on cross-validation performance.
    - best_rfe_model: The RFE model with the optimal number of features.
    """
    # Initialize RandomForest model
    estimator = RandomForestClassifier(random_state=42)

    # Store cross-validation scores for different feature subsets
    mean_scores = []

    # Try different numbers of features (1 to max_features)
    for n_features in range(1, max_features + 1):
        if verbose:
            print(f"Evaluating RFE with {n_features} features...")

        # Set up RFE with the current number of features to select
        rfe = RFE(estimator, n_features_to_select=n_features)

        # Perform cross-validation and store the mean score
        scores = cross_val_score(rfe, X_pca_df, y, cv=5, scoring='accuracy')  # 5-fold CV, accuracy as scoring
        mean_scores.append(np.mean(scores))

        if verbose:
            print(f"Mean cross-validation score with {n_features} features: {np.mean(scores)}")

    # Find the number of features with the highest cross-validation score
    optimal_n_features = np.argmax(mean_scores) + 1  # Add 1 since range starts from 1
    best_rfe_model = RFE(estimator, n_features_to_select=optimal_n_features)
    best_rfe_model.fit(X_pca_df, y)

    if verbose:
        print(f"Optimal number of features: {optimal_n_features}")

    # Return the RFE model with optimal number of features
    return optimal_n_features, best_rfe_model
