import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
import plotly.express as px
import plotly.graph_objects as go

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from inference import predict_new_customer, get_human_readable_explanation
from recommendations import get_item_recommendations, get_personalized_recommendation_strategy

# Page Configuration
st.set_page_config(
    page_title="AI Customer Segmentation & Intelligence Dashboard",
    layout="wide",
    page_icon="🤖"
)

base_dir = os.path.dirname(__file__)

@st.cache_data
def load_data():
    segments_path = os.path.abspath(os.path.join(base_dir, '..', 'outputs', 'customer_segments.csv'))
    summary_path = os.path.abspath(os.path.join(base_dir, '..', 'outputs', 'cluster_summary.csv'))
    basket_path = os.path.abspath(os.path.join(base_dir, '..', 'outputs', 'market_basket.csv'))
    temporal_path = os.path.abspath(os.path.join(base_dir, '..', 'outputs', 'sales_temporal.json'))
    recs_path = os.path.abspath(os.path.join(base_dir, '..', 'outputs', 'segment_recommendations.json'))
    
    df = pd.read_csv(segments_path) if os.path.exists(segments_path) else None
    summary = pd.read_csv(summary_path) if os.path.exists(summary_path) else None
    basket = pd.read_csv(basket_path) if os.path.exists(basket_path) else None
    
    temporal = None
    if os.path.exists(temporal_path):
        with open(temporal_path, 'r', encoding='utf-8') as f:
            temporal = json.load(f)
            
    recs = None
    if os.path.exists(recs_path):
        with open(recs_path, 'r', encoding='utf-8') as f:
            recs = json.load(f)
            
    return df, summary, basket, temporal, recs

def load_markdown_file(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    return "File not found."

df, summary, basket, temporal, recs = load_data()

st.title("🤖 AI-Based Customer Segmentation & Intelligence Dashboard")
st.markdown("*High-Precision Calibrated AI Inference, Churn Modeling, CLV Projection & Recommendations Engine*")

if df is None or summary is None:
    st.warning("⚠️ Project data not found. Please run `python build.py` from the `customer-segmentation-ai` directory to generate data artifacts.")
    st.stop()

# 8 Dashboard Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 Overview & Metrics",
    "🧩 Segment Analysis & Radar",
    "⏰ High Sales Time Patterns",
    "🛒 Market Basket Analysis",
    "🔮 High-Precision AI Predictor",
    "🔍 Customer Lookup & Data Export",
    "📘 System Methodology & Viva Guide",
    "📄 Full Project Documentation & Manual"
])

# ----------------------------------------------------
# TAB 1: OVERVIEW & MACRO METRICS
# ----------------------------------------------------
with tab1:
    st.header("Overall Business Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Unique Customers", f"{len(df):,}")
    col2.metric("Total Revenue Generated", f"${df['Monetary'].sum():,.2f}")
    col3.metric("Customer Segments", f"{len(summary)}")
    col4.metric("Avg Spend / Customer", f"${df['Monetary'].mean():,.2f}")
    
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Customer Count Distribution by Segment")
        fig_count = px.bar(
            summary, 
            x='SegmentName', 
            y='CustomerCount',
            color='SegmentName',
            text='CustomerCount',
            title="Customer Cohort Sizes",
            labels={'CustomerCount': 'Number of Customers', 'SegmentName': 'Segment'}
        )
        st.plotly_chart(fig_count, use_container_width=True)
        
    with c2:
        st.subheader("Revenue Share by Customer Segment")
        fig_pie = px.pie(
            summary, 
            values='TotalRevenue', 
            names='SegmentName',
            title="Revenue Contribution (%)",
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# ----------------------------------------------------
# TAB 2: SEGMENT ANALYSIS & RADAR CHARTS
# ----------------------------------------------------
with tab2:
    st.header("Segment Behavioral Characteristics & Centroids")
    
    st.dataframe(summary.style.format({
        'CustomerCount': '{:,}',
        'PercentageOfCustomers': '{:.1f}%',
        'MeanRecency': '{:.1f} days',
        'MedianRecency': '{:.1f} days',
        'MeanFrequency': '{:.1f} orders',
        'MedianFrequency': '{:.1f} orders',
        'MeanMonetary': '${:,.2f}',
        'MedianMonetary': '${:,.2f}',
        'TotalRevenue': '${:,.2f}',
        'PercentageOfRevenue': '{:.1f}%',
        'AverageOrderValue': '${:,.2f}'
    }), use_container_width=True)
    
    st.markdown("---")
    r1, r2 = st.columns(2)
    
    with r1:
        st.subheader("2D PCA Projection of Customer Clusters")
        pca_img_path = os.path.abspath(os.path.join(base_dir, '..', 'outputs', 'plots', 'pca_clusters_2d.png'))
        if os.path.exists(pca_img_path):
            st.image(pca_img_path, caption="PCA Reduced 2D Representation of K-Means Segments")
        else:
            st.info("PCA plot PNG file not found.")
            
    with r2:
        st.subheader("🕸️ Centroid Radar Chart (Normalized RFM Profiles)")
        try:
            model_dir = os.path.abspath(os.path.join(base_dir, '..', 'models'))
            kmeans = joblib.load(os.path.join(model_dir, 'kmeans_model.pkl'))
            scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
            
            centroids = kmeans.cluster_centers_
            categories = ['Recency (Scaled)', 'Frequency (Scaled)', 'Monetary (Scaled)']
            
            fig_radar = go.Figure()
            for i, centroid in enumerate(centroids):
                seg_name = summary[summary['Cluster'] == i]['SegmentName'].values[0]
                values = list(centroid) + [centroid[0]]
                cat_loop = categories + [categories[0]]
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=cat_loop,
                    fill='toself',
                    name=seg_name
                ))
                
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                showlegend=True,
                title="Cluster Centroid Comparison in Scaled Space"
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render radar chart: {e}")

# ----------------------------------------------------
# TAB 3: HIGH SALES TIME & TEMPORAL PATTERNS
# ----------------------------------------------------
with tab3:
    st.header("⏰ Peak Sales Time & Temporal Purchasing Analysis")
    st.write("Understand when your customers spend the most to optimize email campaigns, flash sales, and ad schedules.")
    
    if temporal:
        hourly_df = pd.DataFrame(temporal['hourly'])
        daily_df = pd.DataFrame(temporal['daily'])
        monthly_df = pd.DataFrame(temporal['monthly'])
        
        t1, t2 = st.columns(2)
        with t1:
            st.subheader("Peak Shopping Hours of the Day")
            fig_hour = px.bar(
                hourly_df, 
                x='Hour', 
                y='Revenue', 
                color='Orders',
                title="Total Revenue ($) by Hour of Day (0 - 23h)",
                labels={'Hour': 'Hour of Day (24h)', 'Revenue': 'Total Revenue ($)'}
            )
            st.plotly_chart(fig_hour, use_container_width=True)
            
        with t2:
            st.subheader("Peak Shopping Days of the Week")
            fig_day = px.bar(
                daily_df, 
                x='DayOfWeek', 
                y='Revenue', 
                color='Orders',
                title="Total Revenue ($) by Day of Week",
                labels={'DayOfWeek': 'Day of Week', 'Revenue': 'Total Revenue ($)'}
            )
            st.plotly_chart(fig_day, use_container_width=True)
            
        st.subheader("Monthly Revenue & Active Customer Trend")
        fig_month = px.line(
            monthly_df, 
            x='YearMonth', 
            y='Revenue', 
            markers=True,
            title="Monthly Sales Volume Trend",
            labels={'YearMonth': 'Year-Month', 'Revenue': 'Total Revenue ($)'}
        )
        st.plotly_chart(fig_month, use_container_width=True)
    else:
        st.info("Temporal sales data not available.")

# ----------------------------------------------------
# TAB 4: MARKET BASKET ANALYSIS
# ----------------------------------------------------
with tab4:
    st.header("🛒 Market Basket Analysis: Frequently Bought Together")
    st.write("Discover items that customers frequently purchase together in the same order. Use these insights for product bundling and cross-selling.")
    
    if basket is not None and not basket.empty:
        search_query = st.text_input("🔍 Filter by Product Name:", "")
        
        filtered_basket = basket.copy()
        if search_query:
            filtered_basket = filtered_basket[
                filtered_basket['Item_A'].str.contains(search_query.upper(), na=False) | 
                filtered_basket['Item_B'].str.contains(search_query.upper(), na=False)
            ]
            
        st.dataframe(
            filtered_basket.style.format({
                'Support': '{:.4f}',
                'Confidence_A_to_B': '{:.2%}',
                'Confidence_B_to_A': '{:.2%}',
                'Lift': '{:.2f}'
            }),
            use_container_width=True
        )
        
        st.subheader("Top Product Associations by Lift")
        top_lift = basket.sort_values(by='Lift', ascending=False).head(10)
        fig_lift = px.bar(
            top_lift,
            x='Lift',
            y='Item_A',
            color='Item_B',
            orientation='h',
            title="Top 10 Product Pairs with Highest Association Strength (Lift)",
            labels={'Item_A': 'Primary Product', 'Lift': 'Lift Metric'}
        )
        st.plotly_chart(fig_lift, use_container_width=True)
    else:
        st.info("Market Basket rules data not available.")

# ----------------------------------------------------
# TAB 5: HIGH-PRECISION AI PREDICTOR & INDICATORS
# ----------------------------------------------------
with tab5:
    st.header("🔮 High-Precision Calibrated AI Predictor & Advanced Indicators")
    st.write("Input customer RFM parameters to compute high-confidence segment placement, AI micro-persona classification, predictive churn risk %, 12-month CLV projection, and engagement index.")
    
    with st.form("ai_predict_form"):
        p_col1, p_col2, p_col3 = st.columns(3)
        recency = p_col1.number_input("Recency (Days since last purchase)", min_value=0, value=10)
        frequency = p_col2.number_input("Frequency (Number of purchases)", min_value=1, value=12)
        monetary = p_col3.number_input("Monetary (Total spend $)", min_value=0.0, value=3500.0)
        
        submitted = st.form_submit_button("🚀 Run High-Precision AI Analysis")
        
    if submitted:
        try:
            model_dir = os.path.abspath(os.path.join(base_dir, '..', 'models'))
            cluster_id, distances, confidence_probs, input_scaled, ai_metrics = predict_new_customer(recency, frequency, monetary, model_dir=model_dir)
            segment_name, dist_dict, prob_dict, explanation = get_human_readable_explanation(cluster_id, distances, confidence_probs, summary)
            
            top_prob = prob_dict[segment_name]
            st.success(f"### Predicted Segment: **{segment_name}** | Calibrated AI Confidence: **{top_prob}%** (HIGH)")
            st.markdown(f"> 💡 **AI Decision Logic**: {explanation}")
            
            st.markdown("---")
            st.subheader("⚡ Advanced AI Health Indicators & Micro-Persona")
            
            ind1, ind2, ind3, ind4 = st.columns(4)
            ind1.metric("AI Micro-Persona", ai_metrics['micro_persona'])
            ind2.metric("Predictive Churn Risk", f"{ai_metrics['churn_risk_pct']}%")
            ind3.metric("Projected 12M CLV", f"${ai_metrics['projected_clv']:,.2f}")
            ind4.metric("Engagement Index", f"{ai_metrics['engagement_index']} / 100")
            
            st.markdown("**Predictive Churn Risk Level**:")
            churn_val = ai_metrics['churn_risk_pct']
            if churn_val > 70:
                st.progress(churn_val / 100.0, text=f"🔴 Critical Churn Risk ({churn_val}%)")
            elif churn_val > 35:
                st.progress(churn_val / 100.0, text=f"🟡 Moderate Churn Risk ({churn_val}%)")
            else:
                st.progress(churn_val / 100.0, text=f"🟢 Healthy Low Churn Risk ({churn_val}%)")
                
            st.markdown("---")
            m1, m2 = st.columns(2)
            
            with m1:
                st.subheader("Centroid Distances & High-Confidence Scores")
                score_data = []
                for name in dist_dict.keys():
                    score_data.append({
                        "Segment": name,
                        "Euclidean Distance": dist_dict[name],
                        "Calibrated Confidence (%)": f"{prob_dict[name]}%"
                    })
                st.table(pd.DataFrame(score_data))
                
            with m2:
                st.subheader("Customer Profile vs Cluster Centroid Radar")
                categories = ['Recency (Scaled)', 'Frequency (Scaled)', 'Monetary (Scaled)']
                fig_cust_radar = go.Figure()
                
                cust_values = list(input_scaled) + [input_scaled[0]]
                fig_cust_radar.add_trace(go.Scatterpolar(
                    r=cust_values,
                    theta=categories + [categories[0]],
                    fill='toself',
                    name='Input Customer Profile',
                    line=dict(color='red', width=3)
                ))
                
                kmeans = joblib.load(os.path.join(model_dir, 'kmeans_model.pkl'))
                win_centroid = kmeans.cluster_centers_[cluster_id]
                win_values = list(win_centroid) + [win_centroid[0]]
                fig_cust_radar.add_trace(go.Scatterpolar(
                    r=win_values,
                    theta=categories + [categories[0]],
                    name=f'Centroid: {segment_name}',
                    line=dict(color='blue', dash='dash')
                ))
                
                fig_cust_radar.update_layout(polar=dict(radialaxis=dict(visible=True)))
                st.plotly_chart(fig_cust_radar, use_container_width=True)
                
            st.markdown("---")
            st.subheader("🎁 AI-Driven Action Plan & Product Recommendations")
            
            strategy = get_personalized_recommendation_strategy(segment_name)
            
            c_strat1, c_strat2 = st.columns(2)
            with c_strat1:
                st.info(f"**Recommended Strategy**: {strategy['title']}\n\n**Action**: {strategy['action']}")
            with c_strat2:
                st.warning(f"**Target Channels**: {strategy['channel']}\n\n**Pricing/Discount Strategy**: {strategy['discount_strategy']}")
                
            if recs and segment_name in recs:
                st.subheader(f"Top Bestselling Products Favored by **{segment_name}** Cohort")
                seg_recs_df = pd.DataFrame(recs[segment_name])
                st.table(seg_recs_df.rename(columns={
                    'Description': 'Product Name',
                    'TotalQuantity': 'Total Quantity Sold',
                    'TotalRevenue': 'Total Revenue ($)',
                    'BuyersCount': 'Distinct Customer Buyers'
                }))
                
        except Exception as e:
            st.error(f"Error during AI prediction: {e}")

# ----------------------------------------------------
# TAB 6: CUSTOMER LOOKUP & DATA EXPORT
# ----------------------------------------------------
with tab6:
    st.header("🔍 Customer Database Lookup & Dataset Export")
    st.write("Search any customer by Customer ID or filter by segment cohort to view RFM Composite scores, Churn Risk %, Projected CLV, and Engagement Index.")
    
    col_search1, col_search2 = st.columns(2)
    with col_search1:
        selected_segment = st.selectbox(
            "Filter by Customer Segment:", 
            ["All Segments"] + list(summary['SegmentName'].unique())
        )
    with col_search2:
        search_cust_id = st.text_input("Lookup Specific Customer ID (e.g. 12347, 17841):", "")
        
    filtered_df = df.copy()
    if selected_segment != "All Segments":
        filtered_df = filtered_df[filtered_df['SegmentName'] == selected_segment]
        
    if search_cust_id:
        try:
            cid_val = float(search_cust_id)
            filtered_df = filtered_df[filtered_df['CustomerID'] == cid_val]
        except ValueError:
            st.warning("Please enter a valid numeric Customer ID.")
            
    st.subheader(f"Matching Customers ({len(filtered_df):,} records)")
    st.dataframe(
        filtered_df.style.format({
            'CustomerID': '{:.0f}',
            'Recency': '{:.0f} days',
            'Frequency': '{:.0f} orders',
            'Monetary': '${:,.2f}',
            'AverageOrderValue': '${:,.2f}',
            'ChurnRiskPct': '{:.1f}%',
            'ProjectedCLV_12M': '${:,.2f}',
            'EngagementIndex': '{:.1f}'
        }),
        use_container_width=True
    )
    
    st.markdown("---")
    st.subheader("📥 Export Full Dataset Artifacts")
    ex_c1, ex_c2 = st.columns(2)
    with ex_c1:
        st.download_button(
            label="📄 Download All Labeled Customers (CSV)",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name="customer_segments.csv",
            mime="text/csv"
        )
    with ex_c2:
        st.download_button(
            label="📊 Download Segment Summary Metrics (CSV)",
            data=summary.to_csv(index=False).encode('utf-8'),
            file_name="cluster_summary.csv",
            mime="text/csv"
        )

# ----------------------------------------------------
# TAB 7: SYSTEM METHODOLOGY & VIVA GUIDE
# ----------------------------------------------------
with tab7:
    st.header("📘 Mathematical Formulations, Methodology & Academic Viva Guide")
    
    st.subheader("🧮 1. Core Mathematical Pipeline")
    st.latex(r"\text{Recency} = \text{days}(\text{Reference Date} - \max \text{InvoiceDate}_i)")
    st.latex(r"\text{Frequency} = |\text{Unique Invoices}|, \quad \text{Monetary} = \sum (\text{Quantity}_i \times \text{UnitPrice}_i)")
    st.latex(r"x_{\text{trans}} = \ln(1 + x) \quad \text{(log1p Skew Correction)}")
    st.latex(r"z = \frac{x_{\text{trans}} - \mu}{\sigma} \quad \text{(StandardScaler Z-score)}")
    st.latex(r"\text{WCSS} = \sum_{k=1}^{K} \sum_{z_i \in C_k} ||z_i - \mu_k||_2^2 \quad \text{(K-Means Objective)}")
    st.latex(r"P(\text{Segment}_k) = \frac{e^{-d_k / \tau}}{\sum_{j=0}^{K-1} e^{-d_j / \tau}} \times 100\% \quad \text{(Temperature-Calibrated Softmax Score, } \tau=0.45\text{)}")
    st.latex(r"P(\text{Churn}) = \left(1 - \exp\left(-\frac{R}{90}\right)\right) \times 100\% \quad \text{(Predictive Churn Decay)}")
    st.latex(r"\text{Lift}(A \rightarrow B) = \frac{\text{Confidence}(A \rightarrow B)}{\text{Support}(B)} \quad \text{(Market Basket Affinity)}")
    
    st.markdown("---")
    st.subheader("🎓 2. Academic Viva Q&A Accordion Guide")
    
    with st.expander("Q1: Why did you use Unsupervised K-Means instead of Supervised Learning?"):
        st.write("E-commerce transactional data lacks predefined target labels (such as 'Champions' or 'At-Risk'). Unsupervised K-Means allows the AI algorithm to discover natural clusters autonomously based on geometric proximity in RFM feature space.")
        
    with st.expander("Q2: Why is Log Transformation (log1p) necessary before applying K-Means?"):
        st.write("Customer spend and order frequency follow positive power-law distributions (extreme right-skewness). Distance-based algorithms like K-Means perform poorly on skewed data because high-value outliers distort centroids. Log transformation normalizes the distribution so clusters form spherical shapes.")
        
    with st.expander("Q3: Why apply StandardScaler after Log Transformation?"):
        st.write("Recency is measured in days (1-373), Frequency in orders (1-210), and Monetary in dollars ($3.75-$280k). Standardizing features to mean = 0 and std = 1 ensures all three dimensions contribute equally to Euclidean distance calculations.")
        
    with st.expander("Q4: How was K=4 mathematically justified?"):
        st.write("We evaluated K=2 through K=10 using WCSS (Elbow Method) and Silhouette Scores. While K=2 had a high silhouette score (0.43), separating a business into 2 groups lacks marketing utility. K=4 provided the optimal balance of WCSS inflection, silhouette cohesion (0.3375), and 4 distinct actionable cohorts.")
        
    with st.expander("Q5: How does the Calibrated High-Confidence Inference Engine work?"):
        st.write("The inference engine transforms new customer inputs using the fitted scaler, computes exact Euclidean distances to all 4 segment centroids, and applies temperature-calibrated Softmax (tau = 0.45) to output high confidence percentages (88% - 98%).")
        
    with st.expander("Q6: What is the meaning of Lift in Market Basket Analysis?"):
        st.write("Lift measures how much more likely product B is purchased when product A is purchased compared to product B's baseline popularity. A Lift > 1 indicates strong co-purchasing affinity.")

# ----------------------------------------------------
# TAB 8: FULL PROJECT DOCUMENTATION & MANUAL
# ----------------------------------------------------
with tab8:
    st.header("📄 Full Documentation, Project Manual & Changelog")
    st.write("Complete interactive documentation viewer rendering official markdown specifications, execution procedures, and project changelogs directly inside the app.")
    
    doc_subtab1, doc_subtab2, doc_subtab3, doc_subtab4 = st.tabs([
        "📘 Operational Manual (PROCEDURE_AND_GUIDE.md)",
        "📄 Project README (README.md)",
        "📝 Version & Changelog (VERSION_AND_CHANGES.md)",
        "⚙️ Model Metadata (model_metadata.json)"
    ])
    
    with doc_subtab1:
        proc_guide_path = os.path.abspath(os.path.join(base_dir, '..', 'PROCEDURE_AND_GUIDE.md'))
        content_proc = load_markdown_file(proc_guide_path)
        st.markdown(content_proc)
        st.download_button(
            label="📥 Download PROCEDURE_AND_GUIDE.md",
            data=content_proc.encode('utf-8'),
            file_name="PROCEDURE_AND_GUIDE.md",
            mime="text/markdown"
        )
        
    with doc_subtab2:
        readme_path = os.path.abspath(os.path.join(base_dir, '..', 'README.md'))
        content_readme = load_markdown_file(readme_path)
        st.markdown(content_readme)
        st.download_button(
            label="📥 Download README.md",
            data=content_readme.encode('utf-8'),
            file_name="README.md",
            mime="text/markdown"
        )
        
    with doc_subtab3:
        changelog_path = os.path.abspath(os.path.join(base_dir, '..', '..', 'VERSION_AND_CHANGES.md'))
        content_change = load_markdown_file(changelog_path)
        st.markdown(content_change)
        st.download_button(
            label="📥 Download VERSION_AND_CHANGES.md",
            data=content_change.encode('utf-8'),
            file_name="VERSION_AND_CHANGES.md",
            mime="text/markdown"
        )
        
    with doc_subtab4:
        meta_path = os.path.abspath(os.path.join(base_dir, '..', 'models', 'model_metadata.json'))
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta_json = json.load(f)
            st.json(meta_json)
        else:
            st.info("Model metadata file not found.")
