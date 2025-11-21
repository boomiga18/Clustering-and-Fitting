"""
This is the template file for the clustering and fitting assignment.
You will be expected to complete all the sections and
make this a fully working, documented file.
You should NOT change any function, file or variable names,
if they are given to you here.
Make use of the functions presented in the lectures
and ensure your code is PEP-8 compliant, including docstrings.
Fitting should be done with only 1 target variable and 1 feature variable,
likewise, clustering should be done with only 2 variables.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression


def plot_relational_plot(df):
    """Scatter plot: Age vs Minutes Played."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x='Age', y='Min', alpha=0.6)
    ax.set_title('Relational Plot: Age vs Minutes Played')
    plt.xlabel('Age')
    plt.ylabel('Minutes Played')
    plt.tight_layout()
    plt.savefig('relational_plot.png')
    plt.close()
    return


def plot_categorical_plot(df):
    """Bar plot of Top 10 Nations by player count."""
    fig, ax = plt.subplots(figsize=(8, 6))
    top_nations = df['Nation'].value_counts().nlargest(10)
    sns.barplot(x=top_nations.values, y=top_nations.index,
                ax=ax, hue=None, palette='viridis', legend=False)
    ax.set_title('Top 10 Nations by Player Count')
    plt.xlabel('Number of Players')
    plt.ylabel('Nation')
    plt.tight_layout()
    plt.savefig('categorical_plot.png')
    plt.close()
    return


def plot_statistical_plot(df):
    """Correlation heatmap for selected numerical features."""
    fig, ax = plt.subplots(figsize=(8, 6))
    cols = ['Age', 'Min', 'Gls', 'Ast', 'xG']
    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('statistical_plot.png')
    plt.close()
    return


def statistical_analysis(df, col: str):
    """Calculate mean, standard deviation, skewness, and excess kurtosis."""
    data = df[col].dropna()
    mean = data.mean()
    stddev = data.std()
    skew = ss.skew(data)
    excess_kurtosis = ss.kurtosis(data)
    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    """Clean and preprocess dataset."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=['Age', 'Min', 'Gls', 'Ast', 'xG'])
    df = df[df['Min'] > 0]
    print("Data Summary:\n", df.describe())
    print("\nCorrelation Matrix:\n", df[['Age', 'Min', 'Gls', 'Ast', 'xG']].corr())
    return df


def writing(moments, col):
    """Print formatted statistical analysis results."""
    print(f'\nFor the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')

    if moments[2] > 0:
        skewness_type = 'right-skewed'
    elif moments[2] < 0:
        skewness_type = 'left-skewed'
    else:
        skewness_type = 'not skewed'

    if moments[3] > 0:
        kurtosis_type = 'leptokurtic'
    elif moments[3] < 0:
        kurtosis_type = 'platykurtic'
    else:
        kurtosis_type = 'mesokurtic'

    print(f'The data was {skewness_type} and {kurtosis_type}.\n')
    return


def perform_clustering(df, col1, col2):
    """Perform K-Means clustering and return labels and centers."""

    def plot_elbow_method(X_scaled):
        """Plot Elbow Method."""
        inertias = []
        K_range = range(2, 7)
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)

        fig, ax = plt.subplots()
        plt.plot(K_range, inertias, 'bo-')
        plt.title('Elbow Method for Optimal K')
        plt.xlabel('Number of clusters (K)')
        plt.ylabel('Inertia')
        plt.tight_layout()
        plt.savefig('elbow_plot.png')
        plt.close()
        return

    def one_silhouette_inertia(X_scaled, k=3):
        """Compute one silhouette score and inertia."""
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(X_scaled)
        _score = silhouette_score(X_scaled, labels)
        _inertia = model.inertia_
        return _score, _inertia

    X = df[[col1, col2]].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    one_silhouette_inertia(X_scaled)
    plot_elbow_method(X_scaled)

    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    centers = kmeans.cluster_centers_

    xkmeans, ykmeans = X_scaled[:, 0], X_scaled[:, 1]
    cenlabels = range(1, 4)

    return labels, X_scaled, xkmeans, ykmeans, cenlabels


def plot_clustered_data(labels, data, xkmeans, ykmeans, centre_labels):
    """Plot clustered data."""
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.title('K-Means Clustering: Goals vs Assists')
    plt.xlabel('Goals (scaled)')
    plt.ylabel('Assists (scaled)')
    plt.tight_layout()
    plt.savefig('clustering.png')
    plt.close()
    return


def perform_fitting(df, col1, col2):
    """Perform simple linear regression: predict col2 from col1."""
    X = df[[col1]].values
    y = df[col2].values
    model = LinearRegression()
    model.fit(X, y)
    x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred = model.predict(x_range)
    return df[[col1, col2]], x_range, y_pred


def plot_fitted_data(data, x, y):
    """Plot regression fit."""
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], color='gray', alpha=0.6)
    plt.plot(x, y, color='red', linewidth=2)
    plt.title('Linear Regression: Expected Goals vs Goals')
    plt.xlabel('Expected Goals (xG)')
    plt.ylabel('Goals (Gls)')
    plt.tight_layout()
    plt.savefig('fitting.png')
    plt.close()
    return


def main():
    """Main function to execute all analyses."""
    df = pd.read_csv('All_Players_1992-2025.csv')
    df = preprocessing(df)
    col = 'Age'
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)
    moments = statistical_analysis(df, col)
    writing(moments, col)
    clustering_results = perform_clustering(df, 'Gls', 'Ast')
    plot_clustered_data(*clustering_results)
    fitting_results = perform_fitting(df, 'xG', 'Gls')
    plot_fitted_data(*fitting_results)

    print("\nFinished. All outputs successfully generated.")
    print("--------------------------------------------------")
    print("Generated plots: relational_plot.png, categorical_plot.png, "
          "statistical_plot.png, elbow_plot.png, clustering.png, fitting.png")
    return


if __name__ == '__main__':
    main()
