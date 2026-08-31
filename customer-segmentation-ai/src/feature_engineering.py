import pandas as pd
import numpy as np

def compute_rfm_features(df):
    """
    Compute Recency, Frequency, and Monetary (RFM) features for each customer.
    """
    print("Computing RFM features...")
    ref_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (ref_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalAmount': 'sum'
    }).reset_index()
    
    rfm.rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'TotalAmount': 'Monetary'
    }, inplace=True)
    
    print(f"Generated RFM features for {len(rfm)} customers.")
    return rfm

def compute_additional_features(df, rfm_df):
    """
    Compute additional behavioral features, composite RFM scores, Churn Risk, CLV, and Engagement Index.
    """
    print("Computing additional behavioral features & AI indicators...")
    
    # Average Order Value (AOV)
    rfm_df['AverageOrderValue'] = rfm_df['Monetary'] / rfm_df['Frequency']
    
    # Quantile scoring for RFM (1 to 5)
    rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm_df['M_Score'] = pd.qcut(rfm_df['Monetary'], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    
    # Composite RFM Score String (e.g. 555)
    rfm_df['RFM_Composite'] = rfm_df['R_Score'].astype(str) + rfm_df['F_Score'].astype(str) + rfm_df['M_Score'].astype(str)
    
    # Predictive Churn Risk (%)
    rfm_df['ChurnRiskPct'] = (1.0 - np.exp(-rfm_df['Recency'] / 90.0)) * 100.0
    rfm_df['ChurnRiskPct'] = rfm_df['ChurnRiskPct'].clip(lower=0.5, upper=99.9).round(1)
    
    # 12-Month Projected Customer Lifetime Value (CLV $)
    annual_freq_est = (rfm_df['Frequency'] * 1.5).clip(upper=52.0)
    retention_prob = (100.0 - rfm_df['ChurnRiskPct']) / 100.0
    rfm_df['ProjectedCLV_12M'] = (rfm_df['AverageOrderValue'] * annual_freq_est * retention_prob).clip(lower=rfm_df['Monetary']).round(2)
    
    # Engagement Index (0 - 100)
    recency_health = 100.0 / (1.0 + 0.015 * rfm_df['Recency'])
    freq_boost = 12.0 * np.log1p(rfm_df['Frequency'])
    rfm_df['EngagementIndex'] = (recency_health + freq_boost).clip(lower=0.0, upper=100.0).round(1)
    
    # Other order metrics
    additional = df.groupby('CustomerID').agg({
        'Quantity': 'sum',
        'StockCode': 'nunique',
        'InvoiceDate': lambda x: x.dt.date.nunique()
    }).reset_index()
    
    additional.rename(columns={
        'Quantity': 'TotalQuantity',
        'StockCode': 'UniqueProducts',
        'InvoiceDate': 'PurchaseDays'
    }, inplace=True)
    
    final_features = pd.merge(rfm_df, additional, on='CustomerID')
    print("Additional features & AI indicators merged.")
    
    return final_features

if __name__ == "__main__":
    pass
