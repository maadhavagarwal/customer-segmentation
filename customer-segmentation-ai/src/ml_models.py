import pandas as pd
import numpy as np
import joblib
import json
import os

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    r2_score,
    mean_absolute_error,
    mean_squared_error
)
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingRegressor,
    IsolationForest
)
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity


def benchmark_clustering_algorithms(X_scaled_df, k=4):
    """
    Benchmarks K-Means against Gaussian Mixture Models (GMM) and Agglomerative
    Hierarchical Clustering using 3 mathematical cluster validation indices:
    - Silhouette Coefficient (Higher is better, [-1, 1])
    - Davies-Bouldin Index (Lower is better, >= 0)
    - Calinski-Harabasz Index / Variance Ratio (Higher is better)
    """
    print("Benchmarking Unsupervised Clustering Algorithms (K-Means vs GMM vs Agglomerative)...")
    algorithms = {
        "K-Means (Lloyd's k-means++)": KMeans(n_clusters=k, init='k-means++', n_init=20, random_state=42),
        "Gaussian Mixture Model (EM)": GaussianMixture(n_components=k, covariance_type='full', random_state=42),
        "Agglomerative Hierarchical (Ward)": AgglomerativeClustering(n_clusters=k, linkage='ward')
    }

    results = []
    for name, model in algorithms.items():
        if hasattr(model, 'fit_predict'):
            labels = model.fit_predict(X_scaled_df)
        else:
            model.fit(X_scaled_df)
            labels = model.predict(X_scaled_df)

        sil = float(silhouette_score(X_scaled_df, labels))
        dbi = float(davies_bouldin_score(X_scaled_df, labels))
        chi = float(calinski_harabasz_score(X_scaled_df, labels))

        results.append({
            "Algorithm": name,
            "Clusters (K)": k,
            "Silhouette_Score": round(sil, 4),
            "Davies_Bouldin_Index": round(dbi, 4),
            "Calinski_Harabasz_Score": round(chi, 2)
        })

    print("Clustering benchmark completed.")
    return results


def train_supervised_churn_and_clv_models(clean_df, rfm_df, model_dir='models'):
    """
    Trains two Supervised Machine Learning models using a temporal observation/target split
    (First 75% observation window -> Last 25% holdout target window):
    1. Random Forest Churn Classifier (Predicts whether customer defects in holdout period)
    2. Gradient Boosting CLV Regressor (Predicts future customer monetary value)
    Also trains a full-dataset Random Forest Churn & CLV model on behavioral features
    for real-time inference on new customer profiles.
    """
    print("Training Supervised ML Models (Random Forest Churn Classifier & Gradient Boosting CLV Regressor)...")
    df = clean_df.copy()
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    min_date = df['InvoiceDate'].min()
    max_date = df['InvoiceDate'].max()
    total_days = (max_date - min_date).days
    cutoff_date = max_date - pd.Timedelta(days=int(total_days * 0.25))

    obs_df = df[df['InvoiceDate'] <= cutoff_date]
    future_df = df[df['InvoiceDate'] > cutoff_date]

    ref_obs = cutoff_date + pd.Timedelta(days=1)
    obs_rfm = obs_df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (ref_obs - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalAmount': 'sum',
        'Quantity': 'sum',
        'StockCode': 'nunique'
    }).reset_index()

    obs_rfm.rename(columns={
        'InvoiceDate': 'Recency_Obs',
        'InvoiceNo': 'Frequency_Obs',
        'TotalAmount': 'Monetary_Obs',
        'Quantity': 'TotalQuantity_Obs',
        'StockCode': 'UniqueProducts_Obs'
    }, inplace=True)
    obs_rfm['AOV_Obs'] = obs_rfm['Monetary_Obs'] / obs_rfm['Frequency_Obs']

    active_future_customers = set(future_df['CustomerID'].unique())
    obs_rfm['Churned_Target'] = obs_rfm['CustomerID'].apply(lambda cid: 0 if cid in active_future_customers else 1)

    feature_cols_obs = ['Recency_Obs', 'Frequency_Obs', 'Monetary_Obs', 'AOV_Obs', 'TotalQuantity_Obs', 'UniqueProducts_Obs']
    X_obs = obs_rfm[feature_cols_obs]
    y_churn = obs_rfm['Churned_Target']

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_obs, y_churn, test_size=0.25, random_state=42, stratify=y_churn
    )

    rf_temporal = RandomForestClassifier(n_estimators=120, max_depth=6, random_state=42)
    rf_temporal.fit(X_train_c, y_train_c)

    y_pred_c = rf_temporal.predict(X_test_c)
    y_prob_c = rf_temporal.predict_proba(X_test_c)[:, 1]

    auc_val = float(roc_auc_score(y_test_c, y_prob_c))
    acc_val = float(accuracy_score(y_test_c, y_pred_c))
    prec_val = float(precision_score(y_test_c, y_pred_c))
    rec_val = float(recall_score(y_test_c, y_pred_c))
    f1_val = float(f1_score(y_test_c, y_pred_c))
    cm = confusion_matrix(y_test_c, y_pred_c).tolist()
    fpr, tpr, _ = roc_curve(y_test_c, y_prob_c)

    clean_feature_names = ['Recency', 'Frequency', 'Monetary', 'AverageOrderValue', 'TotalQuantity', 'UniqueProducts']
    importances = [
        {"Feature": fname, "Importance": round(float(imp), 4)}
        for fname, imp in zip(clean_feature_names, rf_temporal.feature_importances_)
    ]
    importances = sorted(importances, key=lambda x: x['Importance'], reverse=True)

    # Also train an inference-ready Random Forest Classifier & Gradient Boosting CLV Regressor
    # on [Recency, Frequency, Monetary, AverageOrderValue] so Tab 5 can run live ML inference
    infer_features = ['Recency', 'Frequency', 'Monetary', 'AverageOrderValue']
    X_infer_obs = obs_rfm[['Recency_Obs', 'Frequency_Obs', 'Monetary_Obs', 'AOV_Obs']].copy()
    X_infer_obs.columns = infer_features

    rf_churn_live = RandomForestClassifier(n_estimators=120, max_depth=6, random_state=42)
    rf_churn_live.fit(X_infer_obs, y_churn)

    # CLV Regression: Predict 12M Projected CLV from non-target behavioral features
    reg_features = ['Recency', 'Frequency', 'AverageOrderValue', 'TotalQuantity', 'UniqueProducts', 'PurchaseDays']
    X_reg = rfm_df[reg_features]
    y_reg = rfm_df['ProjectedCLV_12M']

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.25, random_state=42)
    gbr_clv = GradientBoostingRegressor(n_estimators=120, max_depth=4, learning_rate=0.08, random_state=42)
    gbr_clv.fit(X_train_r, y_train_r)

    y_pred_r = gbr_clv.predict(X_test_r)
    r2_val = float(r2_score(y_test_r, y_pred_r))
    mae_val = float(mean_absolute_error(y_test_r, y_pred_r))
    rmse_val = float(np.sqrt(mean_squared_error(y_test_r, y_pred_r)))

    reg_importances = [
        {"Feature": fname, "Importance": round(float(imp), 4)}
        for fname, imp in zip(reg_features, gbr_clv.feature_importances_)
    ]
    reg_importances = sorted(reg_importances, key=lambda x: x['Importance'], reverse=True)

    # Save sample Actual vs Predicted for plotting
    sample_idx = np.random.RandomState(42).choice(len(y_test_r), size=min(200, len(y_test_r)), replace=False)
    actual_vs_pred = [
        {"Actual_CLV": round(float(y_test_r.iloc[i]), 2), "Predicted_CLV": round(float(y_pred_r[i]), 2)}
        for i in sample_idx
    ]

    # Live inference CLV Regressor on 4 core features
    gbr_clv_live = GradientBoostingRegressor(n_estimators=120, max_depth=4, learning_rate=0.08, random_state=42)
    gbr_clv_live.fit(rfm_df[infer_features], rfm_df['ProjectedCLV_12M'])

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(rf_churn_live, os.path.join(model_dir, 'rf_churn_model.pkl'))
    joblib.dump(gbr_clv_live, os.path.join(model_dir, 'gbr_clv_model.pkl'))

    # Store ML predicted churn probability into rfm_df
    rfm_df['ML_Churn_Prob_Pct'] = (rf_churn_live.predict_proba(rfm_df[infer_features])[:, 1] * 100.0).round(1)
    rfm_df['ML_Predicted_CLV'] = np.maximum(gbr_clv.predict(rfm_df[reg_features]), rfm_df['Monetary']).round(2)

    supervised_metrics = {
        "churn_classifier": {
            "model_name": "Random Forest Classifier (120 Trees, Temporal Holdout Split)",
            "observation_cutoff_date": str(cutoff_date.date()),
            "train_samples": int(len(X_train_c)),
            "test_samples": int(len(X_test_c)),
            "roc_auc": round(auc_val, 4),
            "accuracy": round(acc_val, 4),
            "precision": round(prec_val, 4),
            "recall": round(rec_val, 4),
            "f1_score": round(f1_val, 4),
            "confusion_matrix": cm,
            "roc_curve": {
                "fpr": [round(float(x), 4) for x in fpr[::max(1, len(fpr)//50)]],
                "tpr": [round(float(x), 4) for x in tpr[::max(1, len(tpr)//50)]]
            },
            "feature_importances": importances
        },
        "clv_regressor": {
            "model_name": "Gradient Boosting Regressor (120 Estimators)",
            "train_samples": int(len(X_train_r)),
            "test_samples": int(len(X_test_r)),
            "r2_score": round(r2_val, 4),
            "mae": round(mae_val, 2),
            "rmse": round(rmse_val, 2),
            "feature_importances": reg_importances,
            "actual_vs_predicted": actual_vs_pred
        }
    }

    print("Supervised ML models trained and saved.")
    return rfm_df, supervised_metrics


def detect_customer_anomalies(rfm_df, model_dir='models'):
    """
    Uses Unsupervised Isolation Forest to detect anomalous purchasing behavior
    (e.g., extreme B2B wholesale bulk buyers or abnormal transaction patterns).
    """
    print("Running Unsupervised Isolation Forest Anomaly Detection...")
    features = ['Recency', 'Frequency', 'Monetary', 'AverageOrderValue']
    X = np.log1p(rfm_df[features])

    iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    preds = iso_forest.fit_predict(X)
    scores = iso_forest.decision_function(X)

    rfm_df['Anomaly_Label'] = np.where(preds == -1, 'Anomaly (Outlier)', 'Normal')
    rfm_df['Anomaly_Score'] = np.round(scores, 4)

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(iso_forest, os.path.join(model_dir, 'isolation_forest.pkl'))

    anomaly_count = int((preds == -1).sum())
    normal_count = int((preds == 1).sum())

    top_anomalies = rfm_df[rfm_df['Anomaly_Label'] == 'Anomaly (Outlier)'].sort_values(
        by='Anomaly_Score', ascending=True
    ).head(15)[['CustomerID', 'SegmentName', 'Recency', 'Frequency', 'Monetary', 'AverageOrderValue', 'Anomaly_Score']].to_dict(orient='records')

    anomaly_summary = {
        "model_name": "Isolation Forest (100 Trees, 5% Contamination)",
        "total_customers": int(len(rfm_df)),
        "normal_customers": normal_count,
        "anomalous_customers": anomaly_count,
        "anomaly_percentage": round((anomaly_count / len(rfm_df)) * 100, 2),
        "top_anomalies": top_anomalies
    }

    print(f"Isolation Forest detected {anomaly_count} anomalous customer profiles.")
    return rfm_df, anomaly_summary


def compute_customer_lookalikes(rfm_df, X_scaled_df, top_k=5):
    """
    Computes Customer-to-Customer Cosine Similarity in standardized 3D RFM space
    and Behavioral 6D space to identify mathematically similar 'Lookalike Customers'.
    Returns a helper dictionary for top customers or can be computed on the fly.
    """
    print("Computing Cosine Similarity Lookalike Matrix...")
    features = ['Recency', 'Frequency', 'Monetary', 'AverageOrderValue', 'UniqueProducts', 'EngagementIndex']
    X_behav = np.log1p(rfm_df[features].clip(lower=0))
    X_norm = (X_behav - X_behav.mean()) / (X_behav.std() + 1e-8)

    sim_matrix = cosine_similarity(X_norm.values)
    customer_ids = rfm_df['CustomerID'].astype(int).values
    segments = rfm_df['SegmentName'].values
    monetaries = rfm_df['Monetary'].values

    # Store top 5 lookalikes for the top 100 customers by spend for fast lookup,
    # while app.py can also compute on-the-fly for any customer ID
    top_indices = np.argsort(-monetaries)[:100]
    lookalike_map = {}

    for idx in top_indices:
        cid = str(int(customer_ids[idx]))
        sims = sim_matrix[idx]
        best_idx = np.argsort(-sims)[1:top_k+1]
        matches = []
        for b_i in best_idx:
            matches.append({
                "Similar_CustomerID": int(customer_ids[b_i]),
                "SegmentName": str(segments[b_i]),
                "Cosine_Similarity": round(float(sims[b_i]), 4),
                "Monetary": round(float(monetaries[b_i]), 2)
            })
        lookalike_map[cid] = matches

    print("Cosine Similarity Lookalike computation completed.")
    return lookalike_map
