import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd
import os

# Set global plot style
sns.set_theme(style="whitegrid", palette="muted")

def plot_elbow_and_silhouette(wcss, silhouette_scores, output_dir='../outputs/plots'):
    """
    Plots the Elbow Curve and Silhouette Score to help select the optimal K.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    k_values = list(wcss.keys())
    wcss_values = list(wcss.values())
    sil_values = list(silhouette_scores.values())
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Elbow curve (Inertia)
    color = 'tab:blue'
    ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax1.set_ylabel('Inertia (WCSS)', color=color, fontsize=12)
    ax1.plot(k_values, wcss_values, marker='o', color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(k_values)
    
    # Silhouette Score
    ax2 = ax1.twinx()  
    color = 'tab:orange'
    ax2.set_ylabel('Silhouette Score', color=color, fontsize=12)  
    ax2.plot(k_values, sil_values, marker='s', color=color, linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Elbow Method and Silhouette Score for Optimal K', fontsize=14, pad=20)
    fig.tight_layout()  
    plt.savefig(os.path.join(output_dir, 'optimal_k_evaluation.png'), dpi=300)
    plt.close()
    
def plot_rfm_distributions(rfm_df, output_dir='../outputs/plots'):
    """
    Plots the distributions (histograms) of Recency, Frequency, and Monetary.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.histplot(rfm_df['Recency'], bins=50, kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title('Recency Distribution')
    
    # Use log scale for F and M to handle skewness visually
    sns.histplot(rfm_df['Frequency'], bins=50, kde=True, ax=axes[1], color='salmon')
    axes[1].set_yscale('log')
    axes[1].set_title('Frequency Distribution (Log Scale)')
    
    sns.histplot(rfm_df['Monetary'], bins=50, kde=True, ax=axes[2], color='lightgreen')
    axes[2].set_yscale('log')
    axes[2].set_title('Monetary Distribution (Log Scale)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rfm_distributions.png'), dpi=300)
    plt.close()

def plot_pca_clusters(X_scaled_df, labels, segment_map=None, output_dir='../outputs/plots'):
    """
    Reduces the scaled features to 2 dimensions using PCA and visualizes the clusters.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled_df)
    
    # Create a DataFrame for plotting
    pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
    
    if segment_map:
        pca_df['Segment'] = [segment_map.get(label, f"Cluster {label}") for label in labels]
    else:
        pca_df['Segment'] = labels
        
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='PCA1', y='PCA2',
        hue='Segment',
        palette=sns.color_palette('hls', len(set(labels))),
        data=pca_df,
        legend='full',
        alpha=0.6,
        s=50
    )
    
    plt.title('2D PCA Visualization of Customer Segments', fontsize=14)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Segment', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca_clusters_2d.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    pass
