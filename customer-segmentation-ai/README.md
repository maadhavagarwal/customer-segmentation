# AI-Based Customer Segmentation Using K-Means Clustering

This project demonstrates the application of Unsupervised Machine Learning to identify distinct customer groups based on their purchasing behavior. It uses the K-Means clustering algorithm on RFM (Recency, Frequency, Monetary) features derived from transactional data.

## Project Structure
```text
customer-segmentation-ai/
│
├── data/                       # Contains the raw 'Online Retail.xlsx' dataset
├── notebooks/                  # Contains the academic step-by-step Jupyter Notebook
├── src/                        # Modular Python scripts for preprocessing, feature engineering, clustering, and evaluation
├── outputs/                    # Output CSVs and generated plots
├── app/                        # Streamlit interactive application
├── requirements.txt            # Python dependencies
└── README.md                   # Project overview
```

## Methodology

1. **Data Preprocessing**: Handling missing values, duplicates, cancellations, and invalid records.
2. **Feature Engineering (RFM Analysis)**: Converting transactional rows into customer-level behavioral features (Recency, Frequency, Monetary).
3. **Data Transformation**: Handling skewness via Log transformation and scaling via `StandardScaler`. This is crucial as K-Means relies on distance metrics.
4. **Optimal K Selection**: Using the Elbow Method (Inertia) and Silhouette Scores to determine the optimal number of clusters.
5. **K-Means Clustering**: Training the AI model and assigning segments.
6. **Interpretation**: Translating mathematical clusters into actionable business segments (e.g., Champions, Loyal Customers, At-Risk Customers).

## How to Run the Project

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Analysis via Jupyter Notebook**:
   Open `notebooks/customer_segmentation.ipynb` in Jupyter Notebook or VS Code to see the step-by-step execution, plots, and insights.
   Running the notebook will also generate the `outputs/` folder containing the clustered data and trained AI models.

3. **Run the Interactive Dashboard (Streamlit)**:
   Ensure you have run the notebook (or the scripts) first so the models exist in `outputs/`. Then launch the app:
   ```bash
   cd app
   streamlit run app.py
   ```
