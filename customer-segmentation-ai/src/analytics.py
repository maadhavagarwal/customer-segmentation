import pandas as pd
import numpy as np
import json
import os
from itertools import combinations
from collections import Counter

def analyze_temporal_sales(clean_df):
    """
    Analyzes temporal purchasing patterns:
    - Sales by hour of day (0-23)
    - Sales by day of week (Monday-Sunday)
    - Monthly revenue trends
    """
    print("Performing temporal sales analysis...")
    df = clean_df.copy()
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Hour of day
    df['Hour'] = df['InvoiceDate'].dt.hour
    hourly_sales = df.groupby('Hour').agg({
        'InvoiceNo': 'nunique',
        'TotalAmount': 'sum',
        'Quantity': 'sum'
    }).reset_index()
    hourly_sales.rename(columns={'InvoiceNo': 'Orders', 'TotalAmount': 'Revenue'}, inplace=True)
    
    # Day of week
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()
    daily_sales = df.groupby('DayOfWeek').agg({
        'InvoiceNo': 'nunique',
        'TotalAmount': 'sum',
        'Quantity': 'sum'
    }).reset_index()
    daily_sales.rename(columns={'InvoiceNo': 'Orders', 'TotalAmount': 'Revenue'}, inplace=True)
    # Ensure correct day order
    daily_sales['DayOfWeek'] = pd.Categorical(daily_sales['DayOfWeek'], categories=days, ordered=True)
    daily_sales = daily_sales.sort_values('DayOfWeek')
    
    # Monthly sales
    df['YearMonth'] = df['InvoiceDate'].dt.to_period('M').astype(str)
    monthly_sales = df.groupby('YearMonth').agg({
        'InvoiceNo': 'nunique',
        'TotalAmount': 'sum',
        'CustomerID': 'nunique'
    }).reset_index()
    monthly_sales.rename(columns={'InvoiceNo': 'Orders', 'TotalAmount': 'Revenue', 'CustomerID': 'ActiveCustomers'}, inplace=True)
    
    temporal_data = {
        'hourly': hourly_sales.to_dict(orient='records'),
        'daily': daily_sales.to_dict(orient='records'),
        'monthly': monthly_sales.to_dict(orient='records')
    }
    
    print("Temporal sales analysis completed.")
    return temporal_data

def market_basket_analysis(clean_df, min_support=0.005, top_n=50):
    """
    Performs Market Basket Analysis (Frequently Bought Together) using invoice co-occurrence.
    Calculates Support, Confidence, and Lift for product pairs.
    """
    print("Performing Market Basket Analysis (Frequently Bought Together)...")
    df = clean_df.copy()
    
    # Filter products with clean descriptions
    df = df.dropna(subset=['Description'])
    df['Description'] = df['Description'].astype(str).str.strip().str.upper()
    
    # Total unique invoices
    total_invoices = df['InvoiceNo'].nunique()
    
    # Group items by invoice (basket)
    baskets = df.groupby('InvoiceNo')['Description'].apply(lambda x: list(set(x))).tolist()
    
    # Filter out single-item baskets for co-occurrence counting
    multi_item_baskets = [basket for basket in baskets if len(basket) > 1]
    
    # Item frequencies
    item_counts = Counter()
    for basket in baskets:
        for item in basket:
            item_counts[item] += 1
            
    # Pair co-occurrences
    pair_counts = Counter()
    for basket in multi_item_baskets:
        # Sort to ensure (A, B) is same as (B, A) for counting
        basket_items = sorted(basket)
        for pair in combinations(basket_items, 2):
            pair_counts[pair] += 1
            
    rules = []
    for (item_a, item_b), count in pair_counts.items():
        support_pair = count / total_invoices
        if support_pair < min_support:
            continue
            
        support_a = item_counts[item_a] / total_invoices
        support_b = item_counts[item_b] / total_invoices
        
        # Rule A -> B
        confidence_a_b = count / item_counts[item_a]
        lift_a_b = confidence_a_b / support_b
        
        # Rule B -> A
        confidence_b_a = count / item_counts[item_b]
        lift_b_a = confidence_b_a / support_a
        
        rules.append({
            'Item_A': item_a,
            'Item_B': item_b,
            'CoOccurrences': count,
            'Support': round(support_pair, 4),
            'Confidence_A_to_B': round(confidence_a_b, 4),
            'Confidence_B_to_A': round(confidence_b_a, 4),
            'Lift': round(lift_a_b, 2)
        })
        
    rules_df = pd.DataFrame(rules)
    if not rules_df.empty:
        rules_df = rules_df.sort_values(by='Lift', ascending=False).head(top_n)
        
    print(f"Generated {len(rules_df)} market basket rules.")
    return rules_df

def analyze_product_performance(clean_df):
    """
    Aggregates top products by revenue and volume sold.
    """
    df = clean_df.copy()
    df['Description'] = df['Description'].astype(str).str.strip().str.upper()
    
    product_stats = df.groupby('Description').agg({
        'Quantity': 'sum',
        'TotalAmount': 'sum',
        'InvoiceNo': 'nunique',
        'CustomerID': 'nunique'
    }).reset_index()
    
    product_stats.rename(columns={
        'Quantity': 'TotalQuantity',
        'TotalAmount': 'TotalRevenue',
        'InvoiceNo': 'TotalOrders',
        'CustomerID': 'UniqueBuyers'
    }, inplace=True)
    
    product_stats = product_stats.sort_values(by='TotalRevenue', ascending=False)
    return product_stats

if __name__ == "__main__":
    pass
