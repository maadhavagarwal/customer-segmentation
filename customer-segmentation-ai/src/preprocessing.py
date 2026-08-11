import pandas as pd
import numpy as np

def load_data(filepath):
    """
    Load the dataset from the given filepath.
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_excel(filepath)
    return df

def clean_data(df):
    """
    Clean the dataset according to the project requirements.
    
    Steps:
    1. Drop rows with missing CustomerID.
    2. Remove duplicate transactions.
    3. Remove cancelled invoices (InvoiceNo starting with 'C').
    4. Remove invalid quantities (<= 0) and unit prices (<= 0).
    5. Calculate TotalAmount.
    """
    print("Starting data cleaning...")
    
    # Initial shape
    initial_shape = df.shape
    print(f"Initial shape: {initial_shape}")
    
    # 1. Drop missing CustomerID
    df = df.dropna(subset=['CustomerID'])
    print(f"Shape after dropping missing CustomerID: {df.shape}")
    
    # 2. Remove duplicates
    df = df.drop_duplicates()
    print(f"Shape after dropping duplicates: {df.shape}")
    
    # 3. Remove cancelled transactions
    # Invoices starting with 'C' are cancellations
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    print(f"Shape after removing cancelled transactions: {df.shape}")
    
    # 4. Remove invalid quantities and unit prices
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    print(f"Shape after removing invalid quantities/prices: {df.shape}")
    
    # 5. Calculate TotalAmount
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    
    print("Data cleaning completed.")
    return df

if __name__ == "__main__":
    df = load_data('../data/Online Retail.xlsx')
    cleaned_df = clean_data(df)
    print(cleaned_df.head())
