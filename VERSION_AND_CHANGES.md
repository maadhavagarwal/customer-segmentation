# Version & Changes Log

This document tracks all major iterations, feature additions, and fixes made to the Customer Segmentation & Multi-Model AI/ML Analytics project.

## [v1.4.0] - Multi-Model AI/ML Suite, Supervised Churn & CLV, XAI Counterfactuals & Isolation Forest
**Date**: October 18, 2026

### Added
- **Multi-Algorithm Clustering Benchmark (`src/ml_models.py`)**: Benchmarked K-Means ($K=4$) against **Gaussian Mixture Models (EM)** and **Agglomerative Hierarchical Clustering (Ward)** across 3 mathematical cluster validation metrics:
  - **Silhouette Coefficient**: K-Means (`0.3373`) vs Agglomerative (`0.2800`) vs GMM (`0.1761`).
  - **Davies-Bouldin Index**: K-Means (`1.0106`) vs Agglomerative (`1.0992`) vs GMM (`1.7048`).
  - **Calinski-Harabasz Score**: K-Means (`3,328.92`) vs Agglomerative (`2,739.57`) vs GMM (`2,170.60`).
- **Supervised Customer Churn Classifier (`src/ml_models.py`, `models/rf_churn_model.pkl`)**: Trained a 120-tree **Random Forest Classifier** using a 75%/25% temporal observation/holdout split (`ROC-AUC = 0.7490`), outputting ROC curves, Confusion Matrix, Precision, Recall, F1-Score, and Gini Feature Importances.
- **Supervised 12-Month CLV Regressor (`src/ml_models.py`, `models/gbr_clv_model.pkl`)**: Trained a 120-estimator **Gradient Boosting Regressor** (`gbr_clv_model.pkl`) to predict customer lifetime value from behavioral features, reporting $R^2$, MAE, RMSE, feature importances, and Actual vs. Predicted log-scatter plots.
- **Unsupervised Anomaly Detection (`src/ml_models.py`, `models/isolation_forest.pkl`)**: Implemented a 100-tree **Isolation Forest** (5% contamination rate) isolating 217 anomalous B2B wholesale/outlier customer profiles from 4,121 normal retail buyers.
- **Local Explainable AI (XAI) & Counterfactual Boundary Solver (`src/inference.py`)**:
  - Decomposes squared Euclidean centroid distance into per-feature percentage penalties across `Recency`, `Frequency`, and `Monetary` z-scores.
  - Solves for the exact minimum counterfactual $\Delta \text{Frequency}$ (orders) and $\Delta \text{Monetary}$ (spend) needed to cross the K-Means decision hyperplane into the **Champions** cluster.
- **Customer Collaborative Filtering (`src/ml_models.py`, `app/app.py`)**: Built a 6D behavioral **Cosine Similarity Lookalike Engine** to find the top 5 mathematically closest peer customers for any target `CustomerID`.
- **9-Tab Streamlit Dashboard (`app/app.py`)**: Added **Tab 6 (AI/ML Model Benchmarks & Lab)** and upgraded **Tab 5 (Segment Predictor & XAI)** and **Tab 7 (Customer Lookup & Similarity)**.


## [v1.3.0] - High-Precision Calibrated Inference, Churn Modeling, CLV & Enterprise Dashboard
**Date**: August 31, 2026

### Added
- **Temperature-Calibrated Softmax Inference (`src/inference.py`)**: Added temperature scaling ($\tau=0.45$) to convert Euclidean centroid distances into calibrated segment probabilities ($88\% - 98\%$ for clear segment matches).
- **Predictive Customer Health Metrics (`src/feature_engineering.py`, `src/inference.py`)**:
  - **Predictive Churn Risk (%)**: Exponential Recency decay model $P(\text{Churn}) = (1 - e^{-R/90}) \times 100\%$.
  - **12-Month Projected CLV ($)**: Forward 12-month spend projection based on Average Order Value, annualized order velocity, and retention probability.
  - **Customer Engagement Index (0–100)**: Composite health score combining Recency decay and logarithmic Frequency boost.
  - **Customer Persona Classification**: Automated assignment of behavioral personas.
  - **RFM Quintile & Composite Scoring**: Added `R_Score`, `F_Score`, `M_Score` (1–5) and composite string (`555` to `111`).
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
- **Explainable Inference Engine (`src/inference.py`)**: Built a robust inference function that strictly enforces the same data transformations used during training.
- **Detailed Segment Analysis**: Updated `src/evaluation.py` to calculate Median Recency, Median Frequency, and Median Monetary, as well as `AverageOrderValue` for each cluster.
- **K-Value Justification**: Computed and logged Inertia and Silhouette scores for K=2 through K=10.
- **Jupyter Notebook Rewrite**: Restructured `notebooks/customer_segmentation.ipynb` into a Viva-ready format.
- **Unit Testing**: Implemented `tests/test_pipeline.py`.


## [v1.0.0] - Initial Release
**Date**: August 11, 2026

### Added
- **Data Preprocessing Pipeline**: Built `src/preprocessing.py` to handle missing `CustomerID` fields, duplicate rows, cancelled invoices, and invalid quantities in the `Online Retail.xlsx` dataset.
- **RFM Feature Engineering**: Developed `src/feature_engineering.py` to aggregate transactional data into customer-level Recency, Frequency, and Monetary (RFM) features, plus Average Order Value.
- **Clustering Logic**: Created `src/clustering.py` using `scikit-learn` to apply `Log1p` transformations and `StandardScaler`.
- **Model Evaluation**: Implemented dynamic logic in `src/evaluation.py` to mathematically interpret cluster centroids and assign human-readable names.
