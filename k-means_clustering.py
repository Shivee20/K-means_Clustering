import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Data loading and EDA
df = pd.read_csv('Frogs_MFCCs.csv')
print(df.shape)
print(df.columns)
print(df.head())
print(df.dtypes)
print(df.isna().sum())
print(df.isnull().sum())
print(df.head())

# Plotting feature distributions
num_cols = 3
num_rows = (len(df.columns) + num_cols - 1) // num_cols
plt.figure(figsize=(15, num_rows * 5))
for i, col in enumerate(df.columns):
    plt.subplot(num_rows, num_cols, i + 1)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Standard scaling
scaler = StandardScaler()
num_columns = df.select_dtypes(include=[np.number]).columns
df[num_columns] = scaler.fit_transform(df[num_columns])
df.drop(columns=['RecordID'], inplace=True)

# Plotting scaled distributions
num_cols = 3
num_rows = (len(df.columns) + num_cols - 1) // num_cols
plt.figure(figsize=(15, num_rows * 5))
for i, col in enumerate(df.columns):
    plt.subplot(num_rows, num_cols, i + 1)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Outlier detection
z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
threshold = 4
outlier_counts = (z_scores > threshold).sum(axis=0)
outlier_summary = pd.DataFrame(outlier_counts, columns=['Number of Outliers'])
outlier_summary.index = df.select_dtypes(include=[np.number]).columns
print(outlier_summary)

# Plot outliers
num_cols = 3
num_rows = (len(df.columns) + num_cols - 1) // num_cols
plt.figure(figsize=(15, num_rows * 5))
for i, col in enumerate(df.select_dtypes(include=[np.number]).columns):
    plt.subplot(num_rows, num_cols, i + 1)
    plt.scatter(df.index, df[col], label='Data Points', color='blue')
    outlier_mask = abs(z_scores[:, i]) > threshold
    outliers = df[col][outlier_mask]
    plt.scatter(df.index[outlier_mask], outliers, label='Outliers', color='red')
    plt.title(f'Feature: {col}')
    plt.xlabel('Index')
    plt.ylabel(col)
    plt.legend()
plt.tight_layout()
plt.show()

# Remove outliers
non_outlier_mask = np.all(z_scores <= threshold, axis=1)
df_cleaned = df[non_outlier_mask].reset_index(drop=True)
print(df.shape)
print(df_cleaned.shape)

# Remove categorical columns
df_cleaned = df_cleaned.select_dtypes(include=[np.number])
print(df_cleaned.head(10))

# Correlation Matrix Heatmap
plt.figure(figsize=(12, 10))
correlation_matrix = df_cleaned.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar=True)
plt.title('Correlation Matrix Heatmap')
plt.show()

# Feature engineering: remove highly correlated features, add their product
threshold = 0.7
correlated_pairs = correlation_matrix[(correlation_matrix > threshold) & (correlation_matrix < 1.0)]
new_columns = {}
columns_to_remove = set()
for i in range(correlated_pairs.shape[0]):
    for j in range(i + 1, correlated_pairs.shape[1]):
        if correlated_pairs.iloc[i, j] > threshold:
            col1 = correlated_pairs.index[i]
            col2 = correlated_pairs.columns[j]
            new_col_name = f"{col1}_x_{col2}"
            new_columns[new_col_name] = df_cleaned[col1] * df_cleaned[col2]
            columns_to_remove.add(col1)
            columns_to_remove.add(col2)
df_cleaned.drop(columns=columns_to_remove, inplace=True)
new_columns_df = pd.DataFrame(new_columns)
df_cleaned = pd.concat([df_cleaned, new_columns_df], axis=1)
print(df_cleaned.shape)
print(df_cleaned.columns.tolist())

# K-Means Clustering: Elbow Method
cluster_range = range(2, 21)
wcss = []
for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(df_cleaned.select_dtypes(include=[np.number]))
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(10, 6))
plt.plot(cluster_range, wcss, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.xticks(cluster_range)
plt.grid()
plt.show()

# Silhouette scores for different clusters
silhouette_scores = []
for optimal_k in range(2, 21):
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(df_cleaned.select_dtypes(include=[np.number]))
    silhouette_avg = silhouette_score(df_cleaned.select_dtypes(include=[np.number]), clusters)
    silhouette_scores.append(silhouette_avg)
plt.figure(figsize=(10, 6))
plt.plot(range(2, 21), silhouette_scores, marker='o', linestyle='-', color='blue')
plt.title('Silhouette Score vs Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.xticks(range(2, 21))
plt.grid()
plt.axhline(y=max(silhouette_scores), color='r', linestyle='--', label='Max Silhouette Score')
plt.legend()
plt.show()

# Compare random and kmeans++ initialization
optimal_k = 6
kmeans_random = KMeans(n_clusters=optimal_k, init='random', random_state=42)
clusters_random = kmeans_random.fit_predict(df_cleaned.select_dtypes(include=[np.number]))
silhouette_avg_random = silhouette_score(df_cleaned.select_dtypes(include=[np.number]), clusters_random)
print(f'Silhouette Score (Random Initialization): {silhouette_avg_random}')
kmeans_plus_plus = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
clusters_plus_plus = kmeans_plus_plus.fit_predict(df_cleaned.select_dtypes(include=[np.number]))
silhouette_avg_plus_plus = silhouette_score(df_cleaned.select_dtypes(include=[np.number]), clusters_plus_plus)
print(f'Silhouette Score (KMeans++ Initialization): {silhouette_avg_plus_plus}')

# PCA for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_cleaned.select_dtypes(include=[np.number]))
df_pca = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
df_pca['Cluster (Random)'] = clusters_random
df_pca['Cluster (KMeans++)'] = clusters_plus_plus
num_clusters = optimal_k
colors = plt.cm.get_cmap('tab10', num_clusters)
color_list = [colors(i) for i in range(num_clusters)]
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
for i in range(num_clusters):
    plt.scatter(df_pca[df_pca['Cluster (Random)'] == i]['PCA1'],
                df_pca[df_pca['Cluster (Random)'] == i]['PCA2'],
                color=color_list[i],
                label=f'Cluster {i}',
                edgecolor='k', alpha=0.6)
plt.title('Clusters with Random Initialization')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.grid(True)
plt.subplot(1, 2, 2)
for i in range(num_clusters):
    plt.scatter(df_pca[df_pca['Cluster (KMeans++)'] == i]['PCA1'],
                df_pca[df_pca['Cluster (KMeans++)'] == i]['PCA2'],
                color=color_list[i],
                label=f'Cluster {i}',
                edgecolor='k', alpha=0.6)
plt.title('Clusters with KMeans++ Initialization')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Feature importance for clustering
centroids_random = kmeans_random.cluster_centers_
mean_data = df_cleaned.select_dtypes(include=[np.number]).mean().values
feature_importance_random = np.abs(centroids_random - mean_data).sum(axis=0)
importance_df_random = pd.DataFrame({
    'Feature': df_cleaned.select_dtypes(include=[np.number]).columns,
    'Importance': feature_importance_random
})
importance_df_random = importance_df_random.sort_values(by='Importance', ascending=False)
print(importance_df_random)
plt.figure(figsize=(12, 6))
plt.barh(importance_df_random['Feature'][:10], importance_df_random['Importance'][:10], color='skyblue')
plt.xlabel('Importance')
plt.title('Top 10 Features Contributing to Cluster Separation (Random Initialization)')
plt.gca().invert_yaxis()
plt.show()

centroids_plus_plus = kmeans_plus_plus.cluster_centers_
mean_data_plus_plus = df_cleaned.select_dtypes(include=[np.number]).mean().values
feature_importance_plus_plus = np.abs(centroids_plus_plus - mean_data_plus_plus).sum(axis=0)
importance_df_plus_plus = pd.DataFrame({
    'Feature': df_cleaned.select_dtypes(include=[np.number]).columns,
    'Importance': feature_importance_plus_plus
})
importance_df_plus_plus = importance_df_plus_plus.sort_values(by='Importance', ascending=False)
print(importance_df_plus_plus)
plt.figure(figsize=(12, 6))
plt.barh(importance_df_plus_plus['Feature'][:10], importance_df_plus_plus['Importance'][:10], color='skyblue')
plt.xlabel('Importance')
plt.title('Top 10 Features Contributing to Cluster Separation (KMeans++ Initialization)')
plt.gca().invert_yaxis()
plt.show()

# Metrics for different cluster numbers
cluster_range = range(2, 21)
silhouette_scores = []
davies_bouldin_scores = []
calinski_harabasz_scores = []
for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    clusters = kmeans.fit_predict(df_cleaned.select_dtypes(include=[np.number]))
    silhouette_avg = silhouette_score(df_cleaned.select_dtypes(include=[np.number]), clusters)
    silhouette_scores.append(silhouette_avg)
    db_index = davies_bouldin_score(df_cleaned.select_dtypes(include=[np.number]), clusters)
    davies_bouldin_scores.append(db_index)
    ch_index = calinski_harabasz_score(df_cleaned.select_dtypes(include=[np.number]), clusters)
    calinski_harabasz_scores.append(ch_index)
metrics_df = pd.DataFrame({
    'Clusters': cluster_range,
    'Silhouette Score': silhouette_scores,
    'Davies-Bouldin Index': davies_bouldin_scores,
    'Calinski-Harabasz Index': calinski_harabasz_scores
})
print(metrics_df)
plt.figure(figsize=(15, 5))
plt.plot(metrics_df['Clusters'], metrics_df['Silhouette Score'], marker='o', label='Silhouette Score', color='blue')
plt.title('Silhouette Score vs. Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.xticks(metrics_df['Clusters'])
plt.legend()
plt.grid()
plt.show()
plt.figure(figsize=(15, 5))
plt.plot(metrics_df['Clusters'], metrics_df['Davies-Bouldin Index'], marker='o', label='Davies-Bouldin Index', color='orange')
plt.title('Davies-Bouldin Index vs. Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Davies-Bouldin Index')
plt.xticks(metrics_df['Clusters'])
plt.legend()
plt.grid()
plt.show()
plt.figure(figsize=(15, 5))
plt.plot(metrics_df['Clusters'], metrics_df['Calinski-Harabasz Index'], marker='o', label='Calinski-Harabasz Index', color='green')
plt.title('Calinski-Harabasz Index vs. Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Calinski-Harabasz Index')
plt.xticks(metrics_df['Clusters'])
plt.legend()
plt.grid()
plt.show()

# Compare clustering algorithms
X = df_cleaned.select_dtypes(include=[np.number])
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
kmeans_clusters = kmeans.fit_predict(X)
agglo = AgglomerativeClustering(n_clusters=optimal_k)
agglo_clusters = agglo.fit_predict(X)
dbscan = DBSCAN()
dbscan_clusters = dbscan.fit_predict(X)
metrics_df = pd.DataFrame({
    'Clustering Method': ['K-Means', 'Agglomerative', 'DBSCAN'],
    'Silhouette Score': [
        silhouette_score(X, kmeans_clusters),
        silhouette_score(X, agglo_clusters),
        silhouette_score(X, dbscan_clusters) if len(set(dbscan_clusters)) > 1 else -1
    ],
    'Davies-Bouldin Index': [
        davies_bouldin_score(X, kmeans_clusters),
        davies_bouldin_score(X, agglo_clusters),
        davies_bouldin_score(X, dbscan_clusters) if len(set(dbscan_clusters)) > 1 else -1
    ],
    'Calinski-Harabasz Index': [
        calinski_harabasz_score(X, kmeans_clusters),
        calinski_harabasz_score(X, agglo_clusters),
        calinski_harabasz_score(X, dbscan_clusters) if len(set(dbscan_clusters)) > 1 else -1
    ]
})
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.bar(metrics_df['Clustering Method'], metrics_df['Silhouette Score'], color='blue')
plt.title('Silhouette Score Comparison')
plt.ylabel('Score')
plt.subplot(1, 3, 2)
plt.bar(metrics_df['Clustering Method'], metrics_df['Davies-Bouldin Index'], color='orange')
plt.title('Davies-Bouldin Index Comparison')
plt.ylabel('Index')
plt.subplot(1, 3, 3)
plt.bar(metrics_df['Clustering Method'], metrics_df['Calinski-Harabasz Index'], color='green')
plt.title('Calinski-Harabasz Index Comparison')
plt.ylabel('Index')
plt.tight_layout()
plt.show()
print(metrics_df)