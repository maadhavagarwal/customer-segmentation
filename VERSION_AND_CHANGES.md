# Version & Changes Log

This document tracks all major iterations, feature additions, and fixes made to the AI Customer Segmentation project.

## [v1.2.0] - Analytics, Market Basket & AI Recommendations
**Date**: August 30, 2026

### Added
- **Market Basket Analysis (`analytics.py`)**: Implemented co-occurrence association analysis on transaction baskets (`InvoiceNo`), generating Support, Confidence, and Lift metrics to discover product pairs frequently bought together.
- **High Sales Time & Temporal Analytics (`analytics.py`)**: Extracted peak purchasing hours of the day (0-23h), peak shopping days of the week, and monthly sales trends.
- **AI Product Recommendations Engine (`recommendations.py`)**: Built segment-based top product recommendation lists as well as item-to-item cross-sell rules. Added personalized next-best-action marketing strategies for each segment cohort.
- **Softmax Confidence Scores & Input Validation (`inference.py`)**: Enhanced the inference engine to compute Softmax probability percentages ($P \propto e^{-d_k/\tau}$) alongside Euclidean centroid distances, enforced strict pandas DataFrame feature schema, and added input validation.
- **Streamlit Dashboard Overhaul (`app/app.py`)**: Expanded Streamlit UI to 5 interactive tabs including Plotly Centroid Radar Charts, peak sales time bar charts, searchable Market Basket Analysis, and AI Recommendation cards.
- **Unit Testing Suite (`tests/test_pipeline.py`)**: Expanded unit tests covering temporal analysis, market basket rules, recommendation strategies, and inference input validation (100% pass rate across 5 test suites).


## [v1.1.0] - Model Explainability & Re-architecture
**Date**: August 11, 2026

### Added
- **Project Restructure**: Created a dedicated `models/` directory for `.pkl` files and `model_metadata.json` to keep output cleaner.
- **Explainable Inference Engine (`inference.py`)**: Built a robust inference function that strictly enforces the same data transformations used during training. It now calculates and displays the Euclidean distance from the new customer's transformed RFM values to all K-Means centroids, providing mathematical transparency on *why* a customer was grouped into a specific segment.
- **Detailed Segment Analysis**: Updated `src/evaluation.py` to calculate Median Recency, Median Frequency, and Median Monetary, as well as `AverageOrderValue` for each cluster. Segments are now assigned strictly via deterministic median ranking rather than arbitrary labeling.
- **K-Value Justification**: Computed and logged Inertia and Silhouette scores for K=2 through K=10. Maintained K=4 due to a spike in silhouette score combined with the business necessity for granular cohorts.
- **Jupyter Notebook Rewrite**: Entirely restructured `notebooks/customer_segmentation.ipynb` into a Viva-ready format detailing the problem statement, algorithm logic, distance metrics, and visualization rationale.
- **Unit Testing**: Implemented `tests/test_pipeline.py` to mathematically verify non-negative RFM outputs, proper duplicate handling, and inference integrity.

### Fixed
- Fixed an issue where the Streamlit UI could cache `None` if models weren't yet fully written to disk.
- Fixed visualization scripts to correctly map updated columns from `evaluation.py`.


## [v1.0.0] - Initial Release
**Date**: August 11, 2026

### Added
- **Data Preprocessing Pipeline**: Built `src/preprocessing.py` to handle missing `CustomerID` fields, duplicate rows, cancelled invoices, and invalid quantities in the `Online Retail.xlsx` dataset.
- **RFM Feature Engineering**: Developed `src/feature_engineering.py` to aggregate transactional data into customer-level Recency, Frequency, and Monetary (RFM) features, plus Average Order Value.
- **AI Clustering Logic**: Created `src/clustering.py` using `scikit-learn` to apply `Log1p` transformations and `StandardScaler`. Tested K=2 through K=10 using WCSS/Inertia and Silhouette Scores, standardizing on **K=4**.
- **Model Evaluation**: Implemented dynamic logic in `src/evaluation.py` to mathematically interpret cluster centroids and assign human-readable names (Champions, Loyal Customers, At-Risk, Lost Customers).
- **Academic Walkthrough**: Generated a comprehensive Jupyter Notebook (`customer_segmentation.ipynb`) detailing the end-to-end Machine Learning methodology suitable for viva/demonstration.
- **Interactive Dashboard**: Designed and deployed a robust `Streamlit` application in `app.py`. Features include:
  - Macro-level business metrics summary.
  - Visual distributions of RFM features and 2D PCA cluster visualizations.
  - **Live Inference Engine**: Ability to input novel RFM values to dynamically scale and predict the customer segment utilizing the pre-trained `kmeans_model.pkl` and `scaler.pkl`.
- **Documentation**: Root `README.md` and this `VERSION_AND_CHANGES.md` tracker created.

### Fixed
- Fixed an initial Streamlit path-resolution bug by utilizing `os.path.dirname(__file__)` to guarantee the app correctly reads the `.pkl` and `.csv` artifacts regardless of the working directory it is launched from.
