import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
import json
import os

def preprocess_for_clustering(rfm_df, features=['Recency', 'Frequency', 'Monetary']):
    """
    Prepares the RFM features for K-Means clustering.
    Because RFM data is typically right-skewed, we apply a log transformation.
    Then we scale the features using StandardScaler.
    """
    print("Preparing features for clustering...")
    X = rfm_df[features].copy()
    
    # Apply log1p transformation to handle skewness
    # log1p is used instead of log to gracefully handle 0s if they exist (though rare in RFM except Monetary=0)
    # However, Monetary can be negative if there were returns, but we removed cancelled invoices and Quantity <= 0.
    X_log = np.log1p(X)
    
    # Initialize scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)
    
    # Create DataFrame for scaled features
    X_scaled_df = pd.DataFrame(X_scaled, columns=[f"{f}_scaled" for f in features], index=X.index)
    
    print("Feature transformation and scaling completed.")
    return X_scaled_df, scaler

def find_optimal_k(X_scaled_df, max_k=10):
    """
    Calculates Inertia (WCSS) and Silhouette Scores for K=2 to max_k.
    Returns a dictionary of metrics.
    """
    print(f"Testing K-Means for K=2 to K={max_k}...")
    wcss = {}
    silhouette_scores = {}
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        kmeans.fit(X_scaled_df)
        
        wcss[k] = kmeans.inertia_
        silhouette_scores[k] = silhouette_score(X_scaled_df, kmeans.labels_)
        print(f"K={k} -> WCSS: {wcss[k]:.2f}, Silhouette Score: {silhouette_scores[k]:.4f}")
        
    return wcss, silhouette_scores

def train_final_model(rfm_df, X_scaled_df, k, scaler, model_dir='../models'):
    """
    Trains the final K-Means model with the selected optimal K.
    Assigns the cluster labels to the original dataframe.
    Saves the model and scaler.
    """
    print(f"Training final K-Means model with K={k}...")
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=20, random_state=42)
    kmeans.fit(X_scaled_df)
    
    # Assign clusters
    rfm_df['Cluster'] = kmeans.labels_
    
    # Save the model, scaler, and metadata
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(kmeans, os.path.join(model_dir, 'kmeans_model.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    
    metadata = {
        "model": "KMeans",
        "algorithm_type": "Unsupervised Learning",
        "selected_k": k,
        "features": ["Recency", "Frequency", "Monetary"],
        "transformation": "log1p",
        "scaling": "StandardScaler",
        "random_state": 42,
        "n_init": 20
    }
    with open(os.path.join(model_dir, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print("Model trained and saved successfully.")
    return rfm_df, kmeans

    pass

if __name__ == "__main__":
    pass
