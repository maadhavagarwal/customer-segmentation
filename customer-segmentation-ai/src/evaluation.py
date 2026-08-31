import pandas as pd

def evaluate_clusters(rfm_df):
    """
    Evaluates the clustered dataset and generates a summary dataframe.
    """
    print("Evaluating clusters...")
    
    total_customers = len(rfm_df)
    total_revenue = rfm_df['Monetary'].sum()
    
    # Calculate Mean and Median for RFM
    cluster_summary = rfm_df.groupby('Cluster').agg({
        'CustomerID': 'count',
        'Recency': ['mean', 'median'],
        'Frequency': ['mean', 'median'],
        'Monetary': ['mean', 'median', 'sum'],
        'AverageOrderValue': 'mean'
    }).reset_index()
    
    # Flatten multi-level columns
    cluster_summary.columns = [
        'Cluster', 'CustomerCount', 
        'MeanRecency', 'MedianRecency', 
        'MeanFrequency', 'MedianFrequency', 
        'MeanMonetary', 'MedianMonetary', 'TotalRevenue',
        'AverageOrderValue'
    ]
    
    # Calculate percentages
    cluster_summary['PercentageOfCustomers'] = (cluster_summary['CustomerCount'] / total_customers) * 100
    cluster_summary['PercentageOfRevenue'] = (cluster_summary['TotalRevenue'] / total_revenue) * 100
    
    # Reorder columns for readability
    cols = ['Cluster', 'CustomerCount', 'PercentageOfCustomers', 
            'MeanRecency', 'MedianRecency', 'MeanFrequency', 'MedianFrequency', 
            'MeanMonetary', 'MedianMonetary', 'TotalRevenue', 'PercentageOfRevenue', 'AverageOrderValue']
    cluster_summary = cluster_summary[cols]
    
    return cluster_summary

def assign_segment_names(cluster_summary):
    """
    Dynamically assigns human-readable segment names based on cluster characteristics.
    Uses Median Recency, Frequency, and Monetary to ensure robustness against outliers.
    """
    print("Assigning segment names based on RFM behavior...")
    
    cluster_summary = cluster_summary.copy()
    
    # Rank clusters based on medians. 
    # Recency: lower is better (ascending=True for rank -> rank 1 is best)
    # Actually, if we use rank(ascending=False) for recency, the lowest recency gets the highest score.
    cluster_summary['R_Score'] = cluster_summary['MedianRecency'].rank(ascending=False)
    # Frequency & Monetary: higher is better
    cluster_summary['F_Score'] = cluster_summary['MedianFrequency'].rank(ascending=True)
    cluster_summary['M_Score'] = cluster_summary['MedianMonetary'].rank(ascending=True)
    
    cluster_summary['Total_Score'] = cluster_summary['R_Score'] + cluster_summary['F_Score'] + cluster_summary['M_Score']
    
    # Sort by total score descending to assign names conceptually
    # Highest score = Champions. Lowest score = Lost Customers.
    # The middle two can be distinguished by Recency vs Frequency.
    
    segment_map = {}
    for i, row in cluster_summary.iterrows():
        cluster_id = int(row['Cluster'])
        r_rank = row['R_Score']
        f_rank = row['F_Score']
        m_rank = row['M_Score']
        total = row['Total_Score']
        
        # Max possible total is 3 * K. For K=4, max is 12.
        if total >= 10:
            name = "Champions"
        elif total <= 5:
            name = "Lost Customers"
        else:
            # Middle tiers
            if r_rank >= 3:
                name = "Loyal Customers" # Good recency and okay F/M
            else:
                name = "At-Risk Customers" # Bad recency but okay F/M
        
        # Fallback to prevent duplicates if scores are tied oddly
        if name in segment_map.values():
            if name == "Loyal Customers": name = "Potential Loyalists"
            elif name == "At-Risk Customers": name = "Hibernating"
            else: name = f"{name} (Tier 2)"
            
        segment_map[cluster_id] = name
            
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
