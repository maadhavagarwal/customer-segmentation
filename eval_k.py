import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import sys
import os

sys.path.append('customer-segmentation-ai/src')
from preprocessing import load_data, clean_data
from feature_engineering import compute_rfm_features
from clustering import preprocess_for_clustering

print("Loading and preparing data...")
df = load_data('customer-segmentation-ai/data/Online Retail.xlsx')
clean_df = clean_data(df)
rfm_df = compute_rfm_features(clean_df)

X_scaled_df, scaler = preprocess_for_clustering(rfm_df, features=['Recency', 'Frequency', 'Monetary'])

print("\nEvaluating K=2 to K=10:")
print(f"{'K':<5} | {'Inertia':<15} | {'Silhouette Score':<15}")
print("-" * 40)

results = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X_scaled_df)
    wcss = kmeans.inertia_
    sil = silhouette_score(X_scaled_df, kmeans.labels_)
    print(f"{k:<5} | {wcss:<15.2f} | {sil:<15.4f}")
    results.append({'K': k, 'WCSS': wcss, 'Silhouette': sil})
