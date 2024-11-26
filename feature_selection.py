from sklearn.decomposition import PCA
import pandas as pd
from constants import log_message, RANDOM_SEED


def PCA_analysis(X: pd.DataFrame, features: str, verbose: bool = True) -> pd.DataFrame:
    """
    Perform Principal Component Analysis (PCA) on the dataset, using Maximum Likelihood Estimation (MLE)
    to automatically determine the optimal number of components.

    Parameters:
    - X (pd.DataFrame): Input DataFrame containing the features and metadata.
    - features (str): Identifier for the feature set, used for saving the loadings.
    - verbose (bool, optional): Flag for enabling verbose logging. Defaults to True.

    Returns:
    - pd.DataFrame: DataFrame containing the PCA-transformed components and original metadata.
    """
    # Exclude non-feature columns (assumes metadata columns; adjust as necessary)
    metadata_columns = ['participant', 'paragraph', 'trial', 'group_id']
    feature_cols = X.drop(columns=metadata_columns, errors='ignore')

    # Initialize PCA with MLE to determine the optimal number of components
    pca = PCA(n_components='mle', random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(feature_cols)

    # Create a DataFrame for PCA-transformed data
    X_pca_df = pd.DataFrame(X_pca, columns=[f'PC{i + 1}' for i in range(pca.n_components_)])

    # Log explained variance ratio and singular values
    if verbose:
        log_message(f"Explained Variance Ratio (first 5 components): {pca.explained_variance_ratio_[:5]}", verbose)
        log_message(f"Singular Values (first 5 components): {pca.singular_values_[:5]}", verbose)
        log_message(f"Transformed PCA Data Shape: {X_pca_df.shape}", verbose)

    # Save PCA component loadings for feature interpretation
    loadings = pd.DataFrame(
        pca.components_,
        columns=feature_cols.columns,
        index=[f'PC{i + 1}' for i in range(pca.n_components_)]
    )
    loadings.to_csv(f"pca_loadings_{features}.csv", index=True)

    # Add original metadata back to the PCA-transformed DataFrame
    X_pca_df['participant'] = X['participant']
    X_pca_df['group_id'] = X['group_id']

    return X_pca_df
