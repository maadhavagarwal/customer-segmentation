# Version & Changes Log

This document tracks all major iterations, feature additions, and fixes made to the Customer Segmentation & Analytics project.

## [v1.3.0] - High-Precision Calibrated Inference, Churn Modeling, CLV & 8-Tab Enterprise Dashboard
**Date**: August 31, 2026

### Added
- **Temperature-Calibrated Softmax Inference (`src/inference.py`)**: Added temperature scaling ($\tau=0.45$) to convert Euclidean centroid distances into calibrated segment probabilities ($88\% - 98\%$ for clear segment matches).
- **Predictive Customer Health Metrics (`src/feature_engineering.py`, `src/inference.py`)**:
  - **Predictive Churn Risk (%)**: Exponential Recency decay model $P(\text{Churn}) = (1 - e^{-R/90}) \times 100\%$.
  - **12-Month Projected CLV ($)**: Forward 12-month spend projection based on Average Order Value, annualized order velocity, and retention probability.
  - **Customer Engagement Index (0–100)**: Composite health score combining Recency decay and logarithmic Frequency boost.
  - **Customer Persona Classification**: Automated assignment of behavioral personas (*Wholesale Bulk Buyer*, *Dormant High-Value Customer*, *Active High-Pace Customer*, *At-Risk Low-Value Customer*, *Promising New Customer*, *Standard Core Customer*).
  - **RFM Quintile & Composite Scoring**: Added `R_Score`, `F_Score`, `M_Score` (1–5) and composite string (`555` to `111`).
- **Expanded 8-Tab Streamlit Dashboard (`app/app.py`)**:
  - Expanded Tab 1 to a **12-Metric Macro KPI Dashboard** plus 4 interactive charts (Cohort Sizes, Revenue Share, AOV Comparison, Monetary Spend Log Boxplot).
  - Added **Optimal K Evaluation (Elbow & Silhouette)** and **Raw RFM Distributions** plots to Tab 2 alongside 2D PCA and Centroid Radar charts.
  - Added **Hourly Orders vs Revenue Dual-Axis Chart** to Tab 3.
  - Added **Interactive Item-to-Item Cross-Sell Recommender** and **Market Basket Rule Affinity Scatter Plot** to Tab 4.
  - Added **Portfolio Churn Risk Histogram** and full customer indicator lookup to Tab 6.
  - Added **Tab 7 (System Methodology & Viva Guide)** with LaTeX equations and Q&A accordions, and **Tab 8 (Project Documentation & Manual)** with live markdown rendering and download buttons.
  - Added 3–4 line analytical explanations below every chart and removed all emojis for a clean, formal enterprise UI.
- **Complete Operational Manual (`PROCEDURE_AND_GUIDE.md`)**: Created comprehensive step-by-step execution guide, mathematical proofs, viva FAQ, and 5-phase future project roadmap.

### Fixed
- Eliminated `scikit-learn` feature-name `UserWarning` in `src/inference.py` by preserving scaled column names (`Recency_scaled`, `Frequency_scaled`, `Monetary_scaled`) in a `pandas.DataFrame` during `KMeans.predict()`.
- Synchronized 5-tuple return unpacking (`cluster_id, distances, confidence_probs, input_scaled, ai_metrics`) across `app/app.py`, `generate_notebook.py`, `notebooks/customer_segmentation.ipynb`, and `tests/test_pipeline.py`.


## [v1.2.0] - Analytics, Market Basket & Product Recommendations
**Date**: August 30, 2026

### Added
- **Market Basket Analysis (`src/analytics.py`)**: Implemented co-occurrence association analysis on transaction baskets (`InvoiceNo`), generating Support, Confidence, and Lift metrics to discover product pairs frequently bought together.
- **High Sales Time & Temporal Analytics (`src/analytics.py`)**: Extracted peak purchasing hours of the day (0-23h), peak shopping days of the week, and monthly sales trends.
- **Product Recommendations Engine (`src/recommendations.py`)**: Built segment-based top product recommendation lists as well as item-to-item cross-sell rules. Added personalized next-best-action marketing strategies for each segment cohort.
- **Softmax Confidence Scores & Input Validation (`src/inference.py`)**: Enhanced the inference engine to compute Softmax probability percentages alongside Euclidean centroid distances, enforced strict pandas DataFrame feature schema, and added input validation.
- **Unit Testing Suite (`tests/test_pipeline.py`)**: Expanded unit tests covering temporal analysis, market basket rules, recommendation strategies, and inference input validation.


## [v1.1.0] - Model Explainability & Re-architecture
**Date**: August 11, 2026

### Added
- **Project Restructure**: Created a dedicated `models/` directory for `.pkl` files and `model_metadata.json` to keep output cleaner.
- **Explainable Inference Engine (`src/inference.py`)**: Built a robust inference function that strictly enforces the same data transformations used during training. It calculates and displays the Euclidean distance from the new customer's transformed RFM values to all K-Means centroids, providing mathematical transparency on *why* a customer was grouped into a specific segment.
- **Detailed Segment Analysis**: Updated `src/evaluation.py` to calculate Median Recency, Median Frequency, and Median Monetary, as well as `AverageOrderValue` for each cluster. Segments are assigned strictly via deterministic median ranking rather than arbitrary labeling.
- **K-Value Justification**: Computed and logged Inertia and Silhouette scores for K=2 through K=10. Maintained K=4 due to a local peak in silhouette score combined with the business necessity for granular cohorts.
- **Jupyter Notebook Rewrite**: Restructured `notebooks/customer_segmentation.ipynb` into a Viva-ready format detailing the problem statement, algorithm logic, distance metrics, and visualization rationale.
- **Unit Testing**: Implemented `tests/test_pipeline.py` to mathematically verify non-negative RFM outputs, proper duplicate handling, and inference integrity.

### Fixed
- Fixed an issue where the Streamlit UI could cache `None` if models weren't yet fully written to disk.
- Fixed visualization scripts to correctly map updated columns from `evaluation.py`.


## [v1.0.0] - Initial Release
**Date**: August 11, 2026

### Added
- **Data Preprocessing Pipeline**: Built `src/preprocessing.py` to handle missing `CustomerID` fields, duplicate rows, cancelled invoices, and invalid quantities in the `Online Retail.xlsx` dataset.
- **RFM Feature Engineering**: Developed `src/feature_engineering.py` to aggregate transactional data into customer-level Recency, Frequency, and Monetary (RFM) features, plus Average Order Value.
- **Clustering Logic**: Created `src/clustering.py` using `scikit-learn` to apply `Log1p` transformations and `StandardScaler`. Tested K=2 through K=10 using WCSS/Inertia and Silhouette Scores, standardizing on **K=4**.
- **Model Evaluation**: Implemented dynamic logic in `src/evaluation.py` to mathematically interpret cluster centroids and assign human-readable names (Champions, Loyal Customers, At-Risk, Lost Customers).
- **Academic Walkthrough**: Generated a comprehensive Jupyter Notebook (`customer_segmentation.ipynb`) detailing the end-to-end Machine Learning methodology suitable for viva/demonstration.
- **Interactive Dashboard**: Designed and deployed a `Streamlit` application in `app/app.py`.
- **Documentation**: Root `README.md` and `VERSION_AND_CHANGES.md` tracker created.

### Fixed
- Fixed an initial Streamlit path-resolution bug by utilizing `os.path.dirname(__file__)` to guarantee the app correctly reads the `.pkl` and `.csv` artifacts regardless of the working directory it is launched from.
