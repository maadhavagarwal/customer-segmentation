import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell("# Customer Segmentation Using K-Means Clustering\n\n## 1. Problem Statement\nBusinesses generate massive amounts of transactional data. Without a structured approach to understand customer behavior, marketing efforts become generic and inefficient. Customer segmentation is required to tailor strategies, maximize ROI, and increase retention."),
    
    nbf.v4.new_markdown_cell("## 2. Objective\nOur goal is to utilize **Unsupervised Machine Learning** to automatically discover hidden patterns and distinct customer groups within the dataset, without relying on predefined labels. This allows the algorithm to mathematically determine the optimal segments."),
    
    nbf.v4.new_markdown_cell("## 3. Dataset\nWe are using the `Online Retail.xlsx` dataset, containing transactional records for a UK-based online retailer. Let's load the data."),
    
    nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nimport sys\nimport os\n\nsys.path.append('../src')\nfrom preprocessing import load_data, clean_data\nfrom feature_engineering import compute_rfm_features, compute_additional_features\nfrom clustering import preprocess_for_clustering, find_optimal_k, train_final_model\nfrom evaluation import evaluate_clusters, assign_segment_names, map_segments_to_data\nfrom visualization import plot_elbow_and_silhouette, plot_rfm_distributions, plot_pca_clusters\nfrom inference import predict_new_customer, get_human_readable_explanation\nfrom analytics import analyze_temporal_sales, market_basket_analysis\nfrom recommendations import generate_segment_recommendations\n\n# Load Data\ndf = load_data('../data/Online Retail.xlsx')\ndisplay(df.head())"),
    
    nbf.v4.new_markdown_cell("## 4. Data Preprocessing\nReal-world data is noisy. We clean the data to prevent our model from learning incorrect patterns.\n\n- We drop rows with missing `CustomerID`.\n- We remove duplicate rows.\n- We filter out cancelled transactions (InvoiceNo starts with 'C').\n- We ensure Quantity and UnitPrice are strictly positive."),
    
    nbf.v4.new_code_cell("clean_df = clean_data(df)\ndisplay(clean_df.describe())"),
    
    nbf.v4.new_markdown_cell("## 5. RFM Feature Engineering\nWe convert transactional rows into customer-level behavioral features known as RFM:\n- **Recency**: Days since the customer's last purchase.\n- **Frequency**: Total number of unique purchases.\n- **Monetary**: Total amount spent.\n\nWe also engineer additional features like Average Order Value, Churn Risk %, 12M Projected CLV $, and Engagement Index."),
    
    nbf.v4.new_code_cell("rfm_df = compute_rfm_features(clean_df)\nrfm_df = compute_additional_features(clean_df, rfm_df)\ndisplay(rfm_df.head())"),
    
    nbf.v4.new_markdown_cell("## 6. Why K-Means?\nK-Means is highly efficient for customer segmentation. It partitions data into K distinct non-overlapping subsets by finding cluster centroids that minimize the variance within each cluster."),
    
    nbf.v4.new_markdown_cell("## 7. Why Scaling?\nK-Means relies on distance calculations (Euclidean distance). Since 'Monetary' values are much larger than 'Frequency' values, Monetary would incorrectly dominate the distance calculations. `StandardScaler` ensures all features contribute equally."),
    
    nbf.v4.new_markdown_cell("## 8. Why Log Transformation?\nCustomer purchasing data is highly right-skewed (few customers spend massive amounts). Log transformation normalizes the distribution, allowing K-Means to identify spherical clusters more easily."),
    
    nbf.v4.new_code_cell("X_scaled_df, scaler = preprocess_for_clustering(rfm_df, features=['Recency', 'Frequency', 'Monetary'])\nplot_rfm_distributions(rfm_df, output_dir='../outputs/plots')\nfrom IPython.display import Image\ndisplay(Image(filename='../outputs/plots/rfm_distributions.png'))"),
    
    nbf.v4.new_markdown_cell("## 9. Choosing K\nWe use two mathematical methods to find K:\n1. **Elbow Method (Inertia/WCSS)**: Finds the point of diminishing returns in minimizing within-cluster variance.\n2. **Silhouette Score**: Measures how similar an object is to its own cluster compared to other clusters (-1 to 1)."),
    
    nbf.v4.new_code_cell("wcss, silhouette_scores = find_optimal_k(X_scaled_df, max_k=10)\nplot_elbow_and_silhouette(wcss, silhouette_scores, output_dir='../outputs/plots')\ndisplay(Image(filename='../outputs/plots/optimal_k_evaluation.png'))"),
    
    nbf.v4.new_markdown_cell("### K-Value Justification\nWhile K=2 may have the highest absolute Silhouette score, separating a business into just two segments lacks actionable granularity. The score stabilizes and slightly rises at K=4, providing the optimal trade-off between mathematical cohesion and business interpretability."),
    
    nbf.v4.new_markdown_cell("## 10. K-Means Model Execution\nThe model algorithm works as follows:\n1. Select K initial centroids randomly.\n2. Assign each customer to the nearest centroid.\n3. Recalculate centroids based on assigned customers.\n4. Repeat until convergence (WCSS is minimized)."),
    
    nbf.v4.new_code_cell("rfm_df, kmeans_model = train_final_model(rfm_df, X_scaled_df, k=4, scaler=scaler, model_dir='../models')\ndisplay(rfm_df.head())"),
    
    nbf.v4.new_markdown_cell("## 11. Cluster Results\nWe evaluate the median behaviors of each mathematical cluster to dynamically map them to human-readable segments."),
    
    nbf.v4.new_code_cell("cluster_summary = evaluate_clusters(rfm_df)\ncluster_summary, segment_map = assign_segment_names(cluster_summary)\nrfm_df = map_segments_to_data(rfm_df, segment_map)\n\nos.makedirs('../outputs', exist_ok=True)\nrfm_df.to_csv('../outputs/customer_segments.csv', index=False)\ncluster_summary.to_csv('../outputs/cluster_summary.csv', index=False)\ndisplay(cluster_summary)"),
    
    nbf.v4.new_markdown_cell("## 12. PCA Visualization\nWe use Principal Component Analysis strictly to reduce our scaled 3D feature space to 2D for visual plotting. The actual model was trained on the full original space."),
    
    nbf.v4.new_code_cell("plot_pca_clusters(X_scaled_df, rfm_df['Cluster'], segment_map, output_dir='../outputs/plots')\ndisplay(Image(filename='../outputs/plots/pca_clusters_2d.png'))"),
    
    nbf.v4.new_markdown_cell("## 13. Market Basket Analysis & Temporal Sales Insights\nWe perform co-occurrence association analysis to discover items frequently bought together and extract peak sales hours/days."),
    
    nbf.v4.new_code_cell("temporal_data = analyze_temporal_sales(clean_df)\nbasket_df = market_basket_analysis(clean_df, min_support=0.003, top_n=10)\ndisplay(basket_df.head())"),
    
    nbf.v4.new_markdown_cell("## 14. New Customer Prediction & Calibrated Inference\nUsing the `inference.py` pipeline, we apply identical transformation and scaling to predict segment placement, segment probability, centroid distances, Churn Risk, CLV, and persona classification."),
    
    nbf.v4.new_code_cell("c_id, dists, probs, scaled, ai_metrics = predict_new_customer(recency=15, frequency=12, monetary=400, model_dir='../models')\nseg_name, dist_dict, prob_dict, explanation = get_human_readable_explanation(c_id, dists, probs, cluster_summary)\nprint(f\"Predicted Segment: {seg_name} (Probability: {prob_dict[seg_name]}%)\\n\")\nprint(f\"Customer Persona: {ai_metrics['micro_persona']}\")\nprint(f\"Churn Risk: {ai_metrics['churn_risk_pct']}%\")\nprint(f\"Projected 12M CLV: ${ai_metrics['projected_clv']:,.2f}\")\nprint(f\"Engagement Index: {ai_metrics['engagement_index']} / 100\\n\")\nprint(\"Centroid Distances & Probabilities:\")\nfor k, v in dist_dict.items():\n    print(f\"- {k}: Distance = {v}, Probability = {prob_dict[k]}%\")\nprint(f\"\\nExplanation: {explanation}\")"),
    
    nbf.v4.new_markdown_cell("## 15. Conclusion & Action Plan\nWe successfully implemented an end-to-end Unsupervised Learning & Analytics pipeline that segments customers, discovers product affinity bundles, and recommends personalized next-best actions.")
]

nb['cells'] = cells

os.makedirs('notebooks', exist_ok=True)
nbf.write(nb, 'notebooks/customer_segmentation.ipynb')
print("Notebook generated successfully!")
