import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


def PCA_analysis(data: DataFrame, attention_score: bool):
    variables = data[['Disengage', 'Awareness', 'FMT']]

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(variables)

    pca = PCA(n_components=3)
    pca.fit(scaled_data)
    loadings = pca.components_[0]

    explained_variance = pca.explained_variance_ratio_
    print(f"Explained Variance: {explained_variance}")

    weighted_average = (scaled_data * loadings).sum(axis=1)
    data['Weighted_Average'] = weighted_average
    data.to_csv('weighted_averages.csv', index=False)

    if attention_score:
        overall_attention_score = data.groupby('Participant')['Weighted_Average'].mean().reset_index()
        overall_attention_score.columns = ['Participant', 'Overall_Attention_Score']

        # Create a separate DataFrame for overall attention scores
        attention_score_df = DataFrame(overall_attention_score)

        # Save to CSV
        attention_score_df.to_csv('attention_scores.csv', index=False)
        print(attention_score_df.head())

        return attention_score_df  # Return the DataFrame for further use


def logistic_regression(data):
    X = data[['Disengage', 'Awareness', 'FMT']]
    y = data['TUT']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    print("Logistic Regression Coefficients:")
    print(log_reg.coef_)


def check_attention_tut_correlation(data: DataFrame, attention_scores: DataFrame):
    # Merge the overall attention scores with the original Data
    merged_data = data.merge(attention_scores, on='Participant')

    # Calculate the correlation between Overall_Attention_Score and TUT
    correlation = merged_data[['Overall_Attention_Score', 'TUT']].corr().iloc[0, 1]

    print(f"Correlation between Overall Attention Score and TUT: {correlation}")


# Example usage after running PCA_analysis and getting attention_scores

