# Customer Segmentation Using K-Means Clustering

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen)

An end-to-end Unsupervised Machine Learning and Analytics pipeline for e-commerce customer segmentation using transactional data from `Online Retail.xlsx`. The system computes **RFM (Recency, Frequency, Monetary)** features, corrects skewness via mathematical transformations, fits **K-Means Clustering ($K=4$)**, performs **Market Basket Association Rule Mining (Support, Confidence, Lift)**, extracts **Temporal Peak Sales Trends**, provides an **Explainable Softmax Inference Engine**, and serves interactive insights via an **8-Tab Streamlit Dashboard**.

---

## Complete Procedure & Guide Manual
For full mathematical proofs, viva Q&A guide, file roles reference, and step-by-step procedures, see [`PROCEDURE_AND_GUIDE.md`](file:///c:/Users/techa/OneDrive/Desktop/des/nmims/sem6/ai/project/customer-segmentation-ai/PROCEDURE_AND_GUIDE.md).

---

## Business Problem & Key Objectives

In e-commerce, generic marketing leads to low conversion rates and high churn. This system automatically groups customers into distinct behavioral cohorts without prior manual labeling.

### Key Capabilities:
1. **Automated Data Cleaning**: Filters missing IDs, duplicates, cancellations (`InvoiceNo` starting with `'C'`), and non-positive prices/quantities.
2. **RFM Feature Aggregation**: Aggregates customer Recency (days), Frequency (order count), Monetary spend ($), and Average Order Value.
3. **Data Normalization & Scaling**: Applies $\text{log1p}$ skewness correction and `StandardScaler` $Z$-score normalization.
4. **Mathematical K Selection**: Validates $K=4$ using WCSS (Elbow Method) and Silhouette Analysis.
5. **Explainable Inference**: Predicts segment placement with exact Euclidean centroid distances and **Softmax Confidence Probabilities** ($P \propto e^{-d_k/\tau}$).
6. **Market Basket Analysis**: Identifies products frequently bought together using Support, Confidence, and Lift metrics.
7. **Temporal Purchasing Analytics**: Analyzes peak shopping hours, peak shopping days of the week, and monthly revenue trends.
8. **Interactive Streamlit Dashboard**: 8-tab web app featuring Plotly Centroid Radar Charts, customer ID lookup, and direct CSV dataset exports.

---

## Repository Structure

```text
customer-segmentation-ai/
│
├── app/
│   └── app.py                        # 8-Tab Streamlit Interactive Web Application
├── data/
│   └── Online Retail.xlsx            # E-Commerce transactional dataset
├── models/                           # Machine Learning artifacts
│   ├── kmeans_model.pkl              # Trained K-Means model (K=4)
│   ├── scaler.pkl                    # Fitted StandardScaler transformer
│   └── model_metadata.json           # Model configuration parameters
├── notebooks/
│   └── customer_segmentation.ipynb   # Academic step-by-step Jupyter Notebook
├── outputs/                          # Generated outputs & plots
│   ├── cluster_summary.csv           # Segment aggregated metrics & medians
│   ├── customer_segments.csv         # All 4,338 labeled customer records
│   ├── market_basket.csv             # Product co-purchase affinity rules
│   ├── sales_temporal.json           # Peak hourly, daily, & monthly trends
│   ├── segment_recommendations.json   # Top products & strategies per cohort
│   └── plots/                        # Generated PNG visual plots
├── src/                              # Modular Python source code
│   ├── analytics.py                  # Market basket & temporal sales analysis
│   ├── clustering.py                 # Scaling, K evaluation, & model fitting
│   ├── evaluation.py                 # Dynamic median ranking & segment naming
│   ├── feature_engineering.py        # RFM & behavioral metric calculation
│   ├── inference.py                  # Explainable inference pipeline
│   ├── preprocessing.py              # Data cleaning & cancellation filtering
│   └── visualization.py              # Plotting routines
├── tests/
│   └── test_pipeline.py              # Unit tests for cleaning, RFM, & inference
├── build.py                          # Full automated ML build script
├── generate_notebook.py               # Programmatic notebook generator
├── requirements.txt                  # Project dependencies
├── PROCEDURE_AND_GUIDE.md            # Execution manual & viva guide
└── README.md                         # Detailed project overview
```

---

## Segment Profiles & Business Action Plan

Based on dynamic median ranking ($K=4$), the model segments **4,338 unique customers** as follows:

| Segment | Customers | % Customers | Median Recency | Median Frequency | Median Spend | Revenue Share | Actionable Strategy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Champions** | 709 | 16.3% | 8 days | 10 orders | $3,722.82 | **64.7%** ($5.75M) | VIP rewards, early access, brand ambassador programs |
| **At-Risk** | 1,173 | 27.0% | 55 days | 4 orders | $1,341.65 | **23.8%** ($2.12M) | Personalized win-back offers & reactivation discounts |
| **Lost Customers** | 1,617 | 37.3% | 177 days | 1 order | $298.40 | **6.3%** ($562K) | Automated low-cost email retargeting or clearance |
| **Loyal Customers**| 839 | 19.3% | 17 days | 2 orders | $466.70 | **5.1%** ($455K) | Cross-selling, up-selling & product recommendations |

---

## Installation & Running Procedures

### 1. Environment Setup
```bash
cd customer-segmentation-ai
pip install -r requirements.txt
```

### 2. Run Automated ML Build Pipeline
```bash
python build.py
```

### 3. Run Automated Unit Tests
```bash
python -m unittest discover -s tests
```

### 4. Launch Interactive Streamlit Web Application
```bash
streamlit run app/app.py
```
Open `http://localhost:8501` in your browser.

---

## License
Academic Project - NMIMS Semester 6 Artificial Intelligence Coursework.
