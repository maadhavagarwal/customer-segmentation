import pandas as pd
import json

file_path = "Online Retail.xlsx"

try:
    df = pd.read_excel(file_path)
    
    summary = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
        "num_duplicates": int(df.duplicated().sum()),
        "cancelled_transactions": int(df['InvoiceNo'].astype(str).str.startswith('C').sum()),
        "date_range": [str(df['InvoiceDate'].min()), str(df['InvoiceDate'].max())] if 'InvoiceDate' in df.columns else None,
        "num_customers": int(df['CustomerID'].nunique()) if 'CustomerID' in df.columns else None,
        "num_transactions": int(df['InvoiceNo'].nunique()) if 'InvoiceNo' in df.columns else None
    }
    
    with open("summary.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    print("Exploration complete. See summary.json")
except Exception as e:
    print(f"Error: {e}")
