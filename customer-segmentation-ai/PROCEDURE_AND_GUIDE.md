# Complete Procedure, Execution Guide & System Documentation Manual

This document serves as the authoritative operational manual, mathematical specification, execution guide, and academic viva preparation document for the Customer Segmentation & Analytics System.

---

## Table of Contents
1. [Project Overview & Objectives](#1-project-overview--objectives)
2. [Step-by-Step Execution Procedure](#2-step-by-step-execution-procedure)
3. [Deep Mathematical & Algorithmic Specifications](#3-deep-mathematical--algorithmic-specifications)
4. [Customer Metrics & Persona Definitions](#4-customer-metrics--persona-definitions)
5. [Project Directory & File Roles Reference](#5-project-directory--file-roles-reference)
6. [Academic Viva & Oral Defense FAQ Guide](#6-academic-viva--oral-defense-faq-guide)
7. [Troubleshooting & Maintenance Procedures](#7-troubleshooting--maintenance-procedures)
8. [Future Upgrades & Project Roadmap](#8-future-upgrades--project-roadmap)

---

## 1. Project Overview & Objectives

In modern e-commerce, mass generic marketing leads to poor conversion rates and customer churn. This project implements an Unsupervised Machine Learning Pipeline using K-Means Clustering (K=4), RFM Feature Engineering, Market Basket Association Analysis, Temporal Sales Trend Analysis, and Temperature-Calibrated Softmax Inference on transactional records from `Online Retail.xlsx`.

### Core System Objectives:
* **Automated Data Processing**: Handle missing records, duplicate transactions, and cancelled orders.
* **RFM Feature Engineering**: Quantify customer value through Recency, Frequency, and Monetary spend metrics.
* **Mathematical Normalization**: Mitigate positive skewness using natural log ($\text{log1p}$) and equalize feature scales using `StandardScaler`.
* **Optimal Cohort Selection**: Validate cluster cohesion ($K=4$) using WCSS (Elbow Method) and Silhouette Scores.
* **Explainable Inference**: Predict segment placement with Calibrated Softmax Probabilities, Churn Risk probabilities, 12-Month CLV projections, and customer persona classifications.
* **Market Basket & Temporal Analytics**: Extract product co-purchase affinity rules (Support, Confidence, Lift) and peak shopping hours/days.
* **Interactive Streamlit Web Dashboard**: Provide business teams and evaluators with an 8-tab interactive web interface.

---

## 2. Step-by-Step Execution Procedure

Follow these exact procedures to install, run, test, and deploy the system.

### Step 2.1: Environment Setup
Ensure Python 3.9 or higher is installed on your system.

```bash
# Navigate to the core project directory
cd customer-segmentation-ai

# Install required dependencies
pip install -r requirements.txt
```

### Step 2.2: Execute the Automated ML Build Pipeline
To process raw data, train the K-Means model, run Market Basket Analysis, generate temporal trends, and save all output artifacts:

```bash
python build.py
```
*Expected Output*: Saved `kmeans_model.pkl`, `scaler.pkl`, `customer_segments.csv`, `cluster_summary.csv`, `market_basket.csv`, `sales_temporal.json`, `segment_recommendations.json`, and PNG plot graphics in `outputs/plots/`.

### Step 2.3: Run Automated Unit Tests
To verify data transformations, cleaning logic, analytics, and inference engine integrity:

```bash
python -m unittest discover -s tests
```
*Expected Output*: `Ran 4 tests in 0.204s ... OK`

### Step 2.4: Launch the Streamlit Web Application
To launch the 8-tab interactive dashboard:

```bash
streamlit run app/app.py
```
*Expected Output*: Opens in your default browser at `http://localhost:8501`.

---

## 3. Deep Mathematical & Algorithmic Specifications

### 3.1 Preprocessing Equations
* **Cancellation Filter**: Remove rows where $\text{InvoiceNo}$ starts with `'C'`.
* **Total Monetary Amount**:
  $$\text{TotalAmount}_i = \text{Quantity}_i \times \text{UnitPrice}_i \quad \forall (\text{Quantity}_i > 0, \text{UnitPrice}_i > 0)$$

### 3.2 RFM Calculation
For customer $c$ with transaction set $S_c$:
$$\text{Recency}_c = \text{days}\left( (\max_{j} \text{InvoiceDate}_j + 1\text{ day}) - \max_{i \in S_c} \text{InvoiceDate}_i \right)$$
$$\text{Frequency}_c = |\{ \text{InvoiceNo}_i \mid i \in S_c \}|$$
$$\text{Monetary}_c = \sum_{i \in S_c} \text{TotalAmount}_i$$

### 3.3 Log Transformation & Scaling
To correct positive right-skewness:
$$x_{\text{trans}} = \ln(1 + x)$$

Standardization to zero mean and unit variance:
$$z = \frac{x_{\text{trans}} - \mu_{\text{trans}}}{\sigma_{\text{trans}}}$$

### 3.4 K-Means Objective Function
$$\arg\min_{\mathcal{C}} \sum_{k=1}^{K} \sum_{z_i \in C_k} ||z_i - \mu_k||_2^2$$

### 3.5 Temperature-Calibrated Softmax Probability ($\tau = 0.45$)
For a new customer scaled feature vector $\mathbf{z}_{\text{new}}$:
$$d_k = ||\mathbf{z}_{\text{new}} - \mu_k||_2$$
$$P(\text{Segment}_k) = \frac{e^{-d_k / \tau}}{\sum_{j=0}^{K-1} e^{-d_j / \tau}} \times 100\% \quad (\tau = 0.45)$$

### 3.6 Market Basket Association Rules
For product pair $(A, B)$:
$$\text{Support}(A \rightarrow B) = \frac{\text{Count}(A \cap B)}{N_{\text{invoices}}}$$
$$\text{Confidence}(A \rightarrow B) = \frac{\text{Support}(A \cap B)}{\text{Support}(A)}$$
$$\text{Lift}(A \rightarrow B) = \frac{\text{Confidence}(A \rightarrow B)}{\text{Support}(B)}$$

---

## 4. Customer Metrics & Persona Definitions

### 4.1 Predictive Churn Risk Score (%)
Uses an exponential Recency decay probability model:
$$P(\text{Churn}) = \left( 1 - \exp\left(-\frac{R}{90}\right) \right) \times 100\%$$

### 4.2 12-Month Projected Customer Lifetime Value (CLV $)
$$\text{CLV}_{12m} = AOV \times \min(F \times 1.5, 52) \times \left( \frac{100 - P(\text{Churn})}{100} \right)$$

### 4.3 Customer Engagement Index (0–100)
$$\text{Engagement} = \min\left(100, \max\left(0, \frac{100}{1 + 0.015 \cdot R} + 12 \cdot \ln(1 + F)\right)\right)$$

### 4.4 Customer Persona Rules
* **Wholesale Bulk Buyer**: $M \ge \$3,000, F \ge 8$
* **Dormant High-Value Customer**: $M \ge \$1,500, R > 60\text{ days}$
* **Active High-Pace Customer**: $R \le 20\text{ days}, F \ge 3$
* **At-Risk Low-Value Customer**: $R > 90\text{ days}, M < \$500$
* **Promising New Customer**: $R \le 30\text{ days}, F \le 2$

---

## 5. Project Directory & File Roles Reference

```text
customer-segmentation-ai/
│
├── app/
│   └── app.py                        # 8-Tab Streamlit Web Application
├── data/
│   └── Online Retail.xlsx            # Raw e-commerce transactional dataset
├── models/
│   ├── kmeans_model.pkl              # Serialized K-Means model (K=4)
│   ├── scaler.pkl                    # Fitted StandardScaler transformer
│   └── model_metadata.json           # Model configuration parameters
├── notebooks/
│   └── customer_segmentation.ipynb   # Step-by-step academic Jupyter Notebook
├── outputs/
│   ├── cluster_summary.csv           # Cohort summary statistics & medians
│   ├── customer_segments.csv         # All 4,338 labeled customer records with CLV & Churn
│   ├── market_basket.csv             # Product co-purchase affinity rules
│   ├── sales_temporal.json           # Hourly, daily, & monthly sales trends
│   ├── segment_recommendations.json   # Top products & strategies per segment
│   └── plots/                        # Generated PNG plot visual artifacts
├── src/
│   ├── analytics.py                  # Market basket & temporal sales analysis
│   ├── clustering.py                 # Feature scaling, K evaluation, & model fitting
│   ├── evaluation.py                 # Dynamic median ranking & segment naming
│   ├── feature_engineering.py        # RFM quintiles, Churn Risk %, CLV $, & Engagement
│   ├── inference.py                  # Temperature-calibrated Softmax engine
│   ├── preprocessing.py              # Data cleaning & cancellation filtering
│   └── recommendations.py            # Segment-based product recommendation engine
├── tests/
│   └── test_pipeline.py              # Automated unit test suite
├── build.py                          # Full automated ML build script
├── generate_notebook.py               # Programmatic Jupyter Notebook generator
├── requirements.txt                  # Python package dependencies
└── README.md                         # Detailed project overview
```

---

## 6. Academic Viva & Oral Defense FAQ Guide

### Q1: Why did you choose Unsupervised K-Means over Supervised Learning?
**Answer**: E-commerce customer transactional datasets lack predefined target labels ("Champions", "At-Risk"). Unsupervised K-Means allows the algorithm to mathematically discover natural clusters within the data based on geometric proximity in feature space.

### Q2: Why is Log Transformation ($\text{log1p}$) necessary before applying K-Means?
**Answer**: Monetary spend and purchase frequency are right-skewed following power-law distributions. Since K-Means relies on Euclidean distances, high-value outliers artificially distort cluster centroids. Applying $\ln(1+x)$ normalizes the distributions so clusters form spherical shapes.

### Q3: Why is `StandardScaler` applied after Log Transformation?
**Answer**: Recency (1–373 days) and Monetary spend ($3.75–$280k) exist on completely different numerical scales. Standardizing features to $\mu=0$ and $\sigma=1$ ensures all three RFM dimensions contribute equally to Euclidean distance calculations.

### Q4: How was K=4 mathematically justified?
**Answer**: We evaluated $K=2$ to $K=10$ using WCSS (Elbow Method) and Silhouette Scores. While $K=2$ had a high silhouette score ($0.43$), dividing a business into 2 cohorts lacks marketing utility. $K=4$ showed a distinct elbow inflection point, a local silhouette peak ($0.3375$), and created 4 distinct business cohorts (Champions, At-Risk, Loyal, Lost).

### Q5: How does the Calibrated Inference Engine work?
**Answer**: `inference.py` transforms new customer inputs using the fitted scaler, computes exact Euclidean distances to all 4 segment centroids, and applies temperature-calibrated Softmax ($\tau = 0.45$) to output segment probabilities.

---

## 7. Troubleshooting & Maintenance Procedures

### Problem 1: `PermissionError: [Errno 13]` when running `build.py`
* **Solution**: Ensure `Online Retail.xlsx` is closed in Microsoft Excel or kill background Excel processes (`Stop-Process -Name EXCEL`).

### Problem 2: `ModuleNotFoundError: No module named 'plotly'`
* **Solution**: Run `pip install plotly` or `pip install -r requirements.txt`.

### Problem 3: Data artifacts missing in Streamlit Dashboard
* **Solution**: Run `python build.py` from the `customer-segmentation-ai` directory.

---

## 8. Future Upgrades & Project Roadmap

### Phase 1: Streaming Architecture & Production REST Microservice
* **Real-Time Data Ingestion**: Transition from batch Excel processing to real-time event streaming using Apache Kafka / AWS Kinesis to ingest checkout transactions instantly.
* **FastAPI ML Microservice**: Wrap the inference engine into high-performance RESTful API endpoints (`/api/v1/predict`, `/api/v1/recommend`) with OpenAPI specifications for integration into Shopify or WooCommerce stores.

### Phase 2: Advanced Supervised & Deep Learning Algorithms
* **Gradient Boosted Decision Trees (XGBoost / LightGBM)**: Train supervised models on longitudinal historical customer logs to replace heuristic churn and CLV formulas with trained predictive models.
* **Neural Collaborative Filtering (Item2Vec / Transformer Embeddings)**: Learn deep item vector representations from purchase sequences to deliver personalized product recommendations.
* **Probabilistic & Density Clustering (Gaussian Mixture Models / HDBSCAN)**: Benchmark soft clustering (GMMs) and density-based clustering (HDBSCAN) against K-Means to discover complex micro-cohorts and automate outlier detection.

### Phase 3: MLOps, Model Monitoring & Automated CI/CD
* **Experiment & Model Tracking (MLflow / Weights & Biases)**: Log hyperparameters, silhouette scores, and version control trained model artifacts automatically.
* **Drift Detection (Evidently AI / Great Expectations)**: Continuously monitor feature drift (distribution shifts in RFM features over time) and concept drift to automatically trigger retrain alerts.
* **Containerization & Deployment (Docker / Kubernetes)**: Package the Streamlit app and API microservice into Docker containers deployed with automated Kubernetes scaling.

### Phase 4: Automated Marketing Automation & CRM Integration
* **CRM Webhook Connectors (Klaviyo / HubSpot / Mailchimp)**: Automatically sync customer segment tags (*Champions*, *At-Risk*, *Loyal*) directly into CRM databases to trigger automated email/SMS campaigns.
* **A/B Testing Framework**: Implement dynamic promotion allocation and measure conversion uplift across segment cohorts.

### Phase 5: Generative AI & Natural Language Assistant Integration
* **LLM Marketing Assistant (RAG / LangChain / OpenAI)**: Integrate an LLM assistant directly into the Streamlit app allowing marketing teams to query customer data via natural language (e.g., *"Generate a win-back email template tailored to At-Risk customers who bought paper plates"*).
