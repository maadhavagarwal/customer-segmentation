import pandas as pd

def compute_rfm_features(df):
    """
    Compute Recency, Frequency, and Monetary (RFM) features for each customer.
    
    Recency: Number of days since the most recent purchase.
    Frequency: Number of unique invoices.
    Monetary: Total amount spent.
    """
    print("Computing RFM features...")
    
    # Calculate the reference date for Recency (usually the max date in dataset + 1 day)
    # Using max date + 1 ensures the most recent purchase has a recency of 1 (not 0)
    ref_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (ref_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalAmount': 'sum'
    }).reset_index()
    
    # Rename columns to standard RFM names
    rfm.rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'TotalAmount': 'Monetary'
    }, inplace=True)
    
    print(f"Generated RFM features for {len(rfm)} customers.")
    return rfm

def compute_additional_features(df, rfm_df):
    """
    Compute additional behavioral features and merge them with the RFM dataframe.
    
    Features:
    - Average Order Value (Monetary / Frequency)
    - Total Quantity Purchased
    - Number of Unique Products (StockCodes)
    - Number of Purchase Days
    """
    print("Computing additional behavioral features...")
    
    # Average Order Value (AOV)
    rfm_df['AverageOrderValue'] = rfm_df['Monetary'] / rfm_df['Frequency']
    
    # Other metrics
    additional = df.groupby('CustomerID').agg({
        'Quantity': 'sum',
        'StockCode': 'nunique',
        'InvoiceDate': lambda x: x.dt.date.nunique() # Number of unique purchase days
    }).reset_index()
    
    additional.rename(columns={
        'Quantity': 'TotalQuantity',
        'StockCode': 'UniqueProducts',
        'InvoiceDate': 'PurchaseDays'
    }, inplace=True)
    
    final_features = pd.merge(rfm_df, additional, on='CustomerID')
    print("Additional features merged.")
    
    return final_features

if __name__ == "__main__":
    pass
