import pandas as pd

def evaluate_clusters(rfm_df):
    """
    Evaluates the clustered dataset and generates a summary dataframe.
    """
    print("Evaluating clusters...")
    
    total_customers = len(rfm_df)
    total_revenue = rfm_df['Monetary'].sum()
    
    cluster_summary = rfm_df.groupby('Cluster').agg({
        'CustomerID': 'count',
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'sum']
    }).reset_index()
    
    # Flatten multi-level columns
    cluster_summary.columns = ['Cluster', 'CustomerCount', 'AverageRecency', 'AverageFrequency', 'AverageMonetary', 'TotalRevenue']
    
    # Calculate percentages
    cluster_summary['PercentageOfCustomers'] = (cluster_summary['CustomerCount'] / total_customers) * 100
    cluster_summary['PercentageOfRevenue'] = (cluster_summary['TotalRevenue'] / total_revenue) * 100
    
    return cluster_summary

def assign_segment_names(cluster_summary):
    """
    Dynamically assigns human-readable segment names based on cluster characteristics.
    Uses Recency, Frequency, and Monetary relative averages to determine segments.
    """
    print("Assigning segment names based on RFM behavior...")
    
    # Rank clusters based on criteria
    # High Monetary and Frequency, Low Recency -> Champions
    # High Recency, Low Frequency, Low Monetary -> Lost Customers
    
    # Simple logic based on ranking:
    cluster_summary = cluster_summary.copy()
    
    # We define relative scores (ranks). Lower recency is better (so we invert rank)
    cluster_summary['R_Score'] = cluster_summary['AverageRecency'].rank(ascending=False)
    cluster_summary['F_Score'] = cluster_summary['AverageFrequency'].rank(ascending=True)
    cluster_summary['M_Score'] = cluster_summary['AverageMonetary'].rank(ascending=True)
    
    cluster_summary['Total_Score'] = cluster_summary['R_Score'] + cluster_summary['F_Score'] + cluster_summary['M_Score']
    
    # Sort by total score descending to assign names
    cluster_summary = cluster_summary.sort_values(by='Total_Score', ascending=False).reset_index(drop=True)
    
    # Possible names for K=4 (Assuming K=4 for this logic, but will be mapped safely)
    names = ["Champions", "Loyal Customers", "At-Risk Customers", "Lost Customers"]
    
    segment_map = {}
    for i, row in cluster_summary.iterrows():
        cluster_id = row['Cluster']
        if i < len(names):
            segment_map[cluster_id] = names[i]
        else:
            segment_map[cluster_id] = f"Segment_{i}"
            
    # Drop temporary score columns
    cluster_summary.drop(columns=['R_Score', 'F_Score', 'M_Score', 'Total_Score'], inplace=True)
    
    cluster_summary['SegmentName'] = cluster_summary['Cluster'].map(segment_map)
    print("Segment assignment complete.")
    
    return cluster_summary, segment_map

def map_segments_to_data(rfm_df, segment_map):
    """
    Maps the segment names back to the original customer dataset.
    """
    rfm_df['SegmentName'] = rfm_df['Cluster'].map(segment_map)
    return rfm_df

if __name__ == "__main__":
    pass
