## Feature selection
# First PCA
# Then exhaustive selection to find best subset of components
from sklearn.decomposition import PCA


def PCA_analysis(X, y):
    pca = PCA(n_components='mle')
    new_X = pca.fit_transform(X)
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)
    print(new_X)
