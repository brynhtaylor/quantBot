import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA

# === Config ===
INPUT_PATH = "data/processed/processed_financial_data.csv"
OUTPUT_DIR = "data/clustered"
PLOT_DIR = "plots/clustering_analysis"

# === Setup ===
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# === Load and Prepare Data ===
df = pd.read_csv(INPUT_PATH, low_memory=False)
df = df.iloc[1:]  # remove labeled header row (duplicated as data)
df = df.dropna()

# === Select Features for Clustering ===
clustering_features = [
    'value_score', 'quality_score', 'bm', 'mom_12m', 'reversal_1m',
    'rolling_vol', 'rel_volume', 'pct_from_ma_20d', 'rsi_14d'
]
X = df[clustering_features].astype(float)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Evaluation Dictionary ===
eval_results = {}

# === Helper: Evaluate Clustering ===
def evaluate_clustering(name, labels, X_subset):
    sil = silhouette_score(X_subset, labels)
    ch = calinski_harabasz_score(X_subset, labels)
    db = davies_bouldin_score(X_subset, labels)
    print(f"{name} | Silhouette: {sil:.3f}, CH: {ch:.2f}, DB: {db:.2f}")
    eval_results[name] = {"silhouette": sil, "calinski_harabasz": ch, "davies_bouldin": db}
    return labels

# === Clustering: K-Means (forced 5 clusters for economic interpretability) ===
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans_labels = evaluate_clustering("KMeans", kmeans.fit_predict(X_scaled), X_scaled)
df['cluster_kmeans'] = kmeans_labels

# === Dimensionality Reduction for Plotting ===
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)
df['pca1'] = X_2d[:, 0]
df['pca2'] = X_2d[:, 1]

# === Plot Clusters ===
def plot_clusters(method):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x='pca1', y='pca2', hue=f'cluster_{method}',
        data=df[df[f'cluster_{method}'] != -1],
        palette='tab10', legend='full', s=10
    )
    plt.title(f"{method.upper()} Clustering (PCA-Reduced)")
    plt.savefig(f"{PLOT_DIR}/{method}_clusters.png")
    plt.close()

for method in ["kmeans"]:
    plot_clusters(method)

# === Save Clustered Dataset ===
df.to_csv(f"{OUTPUT_DIR}/clustered_financial_data.csv", index=False)

# === Save Evaluation Metrics ===
eval_df = pd.DataFrame(eval_results).T
print("\nCluster Evaluation Summary:\n", eval_df)
eval_df.to_csv(f"{OUTPUT_DIR}/clustering_evaluation_summary.csv")

# === Save Cluster Statistics ===
cluster_stats = {}
for method in ["kmeans"]:
    valid_clusters = df
    stats = valid_clusters.groupby(f'cluster_{method}')[clustering_features].mean()
    cluster_stats[method] = stats
    stats.to_csv(f"{OUTPUT_DIR}/{method}_cluster_stats.csv")