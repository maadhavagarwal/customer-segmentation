import pandas as pd
import numpy as np
import os
import sys

sys.path.append('src')

from preprocessing import load_data, clean_data
from feature_engineering import compute_rfm_features, compute_additional_features
from clustering import preprocess_for_clustering, find_optimal_k, train_final_model
from evaluation import evaluate_clusters, assign_segment_names, map_segments_to_data
from visualization import plot_elbow_and_silhouette, plot_rfm_distributions, plot_pca_clusters

print("Running test execution...")
df = load_data('data/Online Retail.xlsx')
clean_df = clean_data(df)
rfm_df = compute_rfm_features(clean_df)
rfm_df = compute_additional_features(clean_df, rfm_df)

plot_rfm_distributions(rfm_df, output_dir='outputs/plots')

X_scaled_df, scaler = preprocess_for_clustering(rfm_df, features=['Recency', 'Frequency', 'Monetary'])
wcss, silhouette_scores = find_optimal_k(X_scaled_df, max_k=5) # use 5 to save time
plot_elbow_and_silhouette(wcss, silhouette_scores, output_dir='outputs/plots')

rfm_df, kmeans_model = train_final_model(rfm_df, X_scaled_df, k=4, scaler=scaler, output_dir='outputs')
cluster_summary = evaluate_clusters(rfm_df)
cluster_summary, segment_map = assign_segment_names(cluster_summary)
rfm_df = map_segments_to_data(rfm_df, segment_map)

rfm_df.to_csv('outputs/customer_segments.csv', index=False)
cluster_summary.to_csv('outputs/cluster_summary.csv', index=False)
plot_pca_clusters(X_scaled_df, rfm_df['Cluster'], segment_map, output_dir='outputs/plots')

print("Test execution complete.")
