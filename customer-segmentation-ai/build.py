import os
import sys
import json

sys.path.append('src')

from preprocessing import load_data, clean_data
from feature_engineering import compute_rfm_features, compute_additional_features
from clustering import preprocess_for_clustering, find_optimal_k, train_final_model
from evaluation import evaluate_clusters, assign_segment_names, map_segments_to_data
from visualization import plot_elbow_and_silhouette, plot_rfm_distributions, plot_pca_clusters
from analytics import analyze_temporal_sales, market_basket_analysis
from recommendations import generate_segment_recommendations
from ml_models import (
    benchmark_clustering_algorithms,
    train_supervised_churn_and_clv_models,
    detect_customer_anomalies,
    compute_customer_lookalikes
)

print("==========================================")
print("Building Customer Segmentation & Analytics")
print("==========================================")

df = load_data('data/Online Retail.xlsx')
clean_df = clean_data(df)

# 1. RFM & Behavioral Features
rfm_df = compute_rfm_features(clean_df)
rfm_df = compute_additional_features(clean_df, rfm_df)

# 2. Clustering & Models
os.makedirs('outputs/plots', exist_ok=True)
plot_rfm_distributions(rfm_df, output_dir='outputs/plots')

X_scaled_df, scaler = preprocess_for_clustering(rfm_df, features=['Recency', 'Frequency', 'Monetary'])
wcss, silhouette_scores = find_optimal_k(X_scaled_df, max_k=10)
plot_elbow_and_silhouette(wcss, silhouette_scores, output_dir='outputs/plots')

rfm_df, kmeans_model = train_final_model(rfm_df, X_scaled_df, k=4, scaler=scaler, model_dir='models')
cluster_summary = evaluate_clusters(rfm_df)
cluster_summary, segment_map = assign_segment_names(cluster_summary)
rfm_df = map_segments_to_data(rfm_df, segment_map)

# 3. Advanced AI/ML Benchmarks, Supervised Models, Anomaly Detection & Cosine Similarity
clustering_benchmarks = benchmark_clustering_algorithms(X_scaled_df, k=4)
rfm_df, supervised_metrics = train_supervised_churn_and_clv_models(clean_df, rfm_df, model_dir='models')
rfm_df, anomaly_summary = detect_customer_anomalies(rfm_df, model_dir='models')
lookalike_map = compute_customer_lookalikes(rfm_df, X_scaled_df, top_k=5)

ml_benchmarks_payload = {
    "clustering_benchmarks": clustering_benchmarks,
    "supervised_models": supervised_metrics,
    "anomaly_detection": anomaly_summary,
    "customer_lookalikes": lookalike_map
}

# 4. Analytics & High Sales Time
temporal_data = analyze_temporal_sales(clean_df)
market_basket_df = market_basket_analysis(clean_df, min_support=0.003, top_n=100)

# 5. Segment Product Recommendations
segment_recommendations = generate_segment_recommendations(clean_df, rfm_df, top_n=5)

# Save Output Artifacts
print("Saving output datasets and analytics files...")
os.makedirs('outputs', exist_ok=True)
rfm_df.to_csv('outputs/customer_segments.csv', index=False)
cluster_summary.to_csv('outputs/cluster_summary.csv', index=False)
market_basket_df.to_csv('outputs/market_basket.csv', index=False)

with open('outputs/sales_temporal.json', 'w') as f:
    json.dump(temporal_data, f, indent=4)
    
with open('outputs/segment_recommendations.json', 'w') as f:
    json.dump(segment_recommendations, f, indent=4)

with open('outputs/ml_benchmarks.json', 'w') as f:
    json.dump(ml_benchmarks_payload, f, indent=4)

plot_pca_clusters(X_scaled_df, rfm_df['Cluster'], segment_map, output_dir='outputs/plots')

print("==========================================")
print("Build Pipeline Completed Successfully!")
print("==========================================")
