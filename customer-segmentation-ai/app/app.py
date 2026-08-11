import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import sys

# Add src to path so we can import visualization functions if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from clustering import predict_customer_segment

# Configuration
st.set_page_config(page_title="AI Customer Segmentation", layout="wide", page_icon="📊")

# Load Data and Models
@st.cache_data
def load_app_data():
    segments_path = os.path.join('..', 'outputs', 'customer_segments.csv')
    summary_path = os.path.join('..', 'outputs', 'cluster_summary.csv')
    
    if os.path.exists(segments_path) and os.path.exists(summary_path):
        df = pd.read_csv(segments_path)
        summary = pd.read_csv(summary_path)
        return df, summary
    else:
        return None, None

df, summary = load_app_data()

st.title("🤖 AI-Based Customer Segmentation Dashboard")

if df is None:
    st.warning("Data not found. Please ensure the K-Means model has been trained and outputs generated in the 'outputs' folder.")
    st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs(["Dashboard", "Segment Analysis", "AI Prediction"])

with tab1:
    st.header("Overall Business Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{len(df):,}")
    col2.metric("Total Revenue", f"${df['Monetary'].sum():,.2f}")
    col3.metric("Number of Segments", len(summary))
    col4.metric("Avg Customer Value", f"${df['Monetary'].mean():,.2f}")
    
    st.subheader("Segment Distribution")
    st.bar_chart(summary.set_index('SegmentName')['CustomerCount'])

with tab2:
    st.header("Segment Characteristics")
    st.dataframe(summary.style.format({
        'CustomerCount': '{:,}',
        'AverageRecency': '{:.1f}',
        'AverageFrequency': '{:.1f}',
        'AverageMonetary': '${:,.2f}',
        'TotalRevenue': '${:,.2f}',
        'PercentageOfCustomers': '{:.1f}%',
        'PercentageOfRevenue': '{:.1f}%'
    }))
    
    st.subheader("Visualizations")
    pca_img_path = os.path.join('..', 'outputs', 'plots', 'pca_clusters_2d.png')
    rfm_img_path = os.path.join('..', 'outputs', 'plots', 'rfm_distributions.png')
    k_img_path = os.path.join('..', 'outputs', 'plots', 'optimal_k_evaluation.png')
    
    c1, c2 = st.columns(2)
    if os.path.exists(pca_img_path):
        c1.image(pca_img_path, caption="2D PCA Visualization of Clusters")
    if os.path.exists(rfm_img_path):
        c2.image(rfm_img_path, caption="RFM Feature Distributions")
    if os.path.exists(k_img_path):
        st.image(k_img_path, caption="Elbow Method & Silhouette Score (Finding K)")

with tab3:
    st.header("🔮 Predict Customer Segment")
    st.write("Enter the RFM values for a customer to see which segment the AI model assigns them to.")
    
    with st.form("predict_form"):
        recency = st.number_input("Recency (Days since last purchase)", min_value=0, value=10)
        frequency = st.number_input("Frequency (Number of purchases)", min_value=1, value=5)
        monetary = st.number_input("Monetary (Total amount spent $)", min_value=0.0, value=250.0)
        
        submitted = st.form_submit_button("Predict Segment")
        
        if submitted:
            try:
                cluster_id = predict_customer_segment(
                    recency, frequency, monetary,
                    model_path=os.path.join('..', 'outputs', 'kmeans_model.pkl'),
                    scaler_path=os.path.join('..', 'outputs', 'scaler.pkl')
                )
                
                segment_name = summary[summary['Cluster'] == cluster_id]['SegmentName'].values[0]
                
                st.success(f"**Predicted Segment:** {segment_name} (Cluster {cluster_id})")
                
                # Recommendations based on segment
                if "Champion" in segment_name:
                    st.info("**Strategy**: Loyalty rewards, Exclusive products, Early access.")
                elif "Loyal" in segment_name:
                    st.info("**Strategy**: Cross-selling, Personalized recommendations.")
                elif "Risk" in segment_name:
                    st.info("**Strategy**: Personalized offers, Reminder campaigns, Limited-time discounts.")
                elif "Lost" in segment_name:
                    st.info("**Strategy**: Win-back campaigns, Reactivation offers.")
                else:
                    st.info("**Strategy**: General engagement campaigns.")
                    
            except Exception as e:
                st.error(f"Prediction failed. Ensure models are trained and saved. Error: {e}")
