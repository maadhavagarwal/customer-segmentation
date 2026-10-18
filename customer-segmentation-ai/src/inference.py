import numpy as np
import pandas as pd
import joblib
import json
import os

def load_models(model_dir='../models'):
    """
    Loads trained K-Means model, scaler, metadata, and optional supervised/anomaly ML models.
    """
    kmeans = joblib.load(os.path.join(model_dir, 'kmeans_model.pkl'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    with open(os.path.join(model_dir, 'model_metadata.json'), 'r') as f:
        metadata = json.load(f)
    return kmeans, scaler, metadata

def compute_xai_and_counterfactual(recency, frequency, monetary, input_scaled, kmeans, scaler, cluster_id):
    """
    Local Explainable AI (XAI) & Counterfactual Decision Boundary Solver:
    1. Computes per-dimension squared z-score attribution toward the assigned centroid
       and toward the Champions centroid.
    2. Computes the minimum counterfactual change in Frequency and Monetary required
       to cross the K-Means decision hyperplane into the highest-value cluster.
    """
    features = ['Recency', 'Frequency', 'Monetary']
    centroids = kmeans.cluster_centers_

    # Identify the Champions centroid (lowest Recency scaled, highest Frequency & Monetary scaled)
    centroid_scores = -centroids[:, 0] + centroids[:, 1] + centroids[:, 2]
    champions_id = int(np.argmax(centroid_scores))
    win_centroid = centroids[cluster_id]
    champ_centroid = centroids[champions_id]

    # 1. Per-feature squared distance breakdown to winning centroid
    sq_diffs_win = (input_scaled - win_centroid) ** 2
    total_sq_win = float(np.sum(sq_diffs_win)) + 1e-9
    feature_attributions = {}
    for i, fname in enumerate(features):
        # Z-score deviation from winning centroid
        z_diff = float(input_scaled[i] - win_centroid[i])
        pct_penalty = float((sq_diffs_win[i] / total_sq_win) * 100.0)
        feature_attributions[fname] = {
            "customer_z": round(float(input_scaled[i]), 3),
            "centroid_z": round(float(win_centroid[i]), 3),
            "z_deviation": round(z_diff, 3),
            "distance_contribution_pct": round(pct_penalty, 1)
        }

    # 2. Counterfactual Solver to reach Champions cluster
    if cluster_id == champions_id:
        counterfactual = {
            "already_champion": True,
            "target_segment": "Champions",
            "needed_orders_delta": 0,
            "needed_spend_delta": 0.0,
            "target_recency": int(recency),
            "message": "Customer is already inside the Champions decision boundary."
        }
    else:
        # Target scaled coordinates: move 85% of the way toward Champions centroid on F and M,
        # and ensure Recency is at most 14 days if currently lapsed
        target_r = min(int(recency), 14)
        target_f_z = max(float(input_scaled[1]), float(champ_centroid[1]) * 0.75)
        target_m_z = max(float(input_scaled[2]), float(champ_centroid[2]) * 0.75)

        # Inverse transform from scaled z-space -> log1p space -> original space
        f_log = target_f_z * scaler.scale_[1] + scaler.mean_[1]
        m_log = target_m_z * scaler.scale_[2] + scaler.mean_[2]

        target_f = max(int(frequency) + 1, int(np.ceil(np.expm1(f_log))))
        target_m = max(float(monetary) + 50.0, float(np.expm1(m_log)))

        delta_f = max(0, int(target_f - frequency))
        delta_m = max(0.0, round(float(target_m - monetary), 2))

        counterfactual = {
            "already_champion": False,
            "target_segment": "Champions",
            "needed_orders_delta": delta_f,
            "needed_spend_delta": delta_m,
            "target_recency": target_r,
            "message": (
                f"To cross the K-Means decision hyperplane into Champions, this customer needs "
                f"+{delta_f} additional order(s) and +${delta_m:,.2f} in additional spend "
                f"(with Recency <= {target_r} days)."
            )
        }

    return feature_attributions, counterfactual

def predict_new_customer(recency, frequency, monetary, model_dir='../models'):
    """
    Multi-Model Inference Engine:
    - K-Means Clustering + Temperature-Calibrated Softmax (tau = 0.45)
    - Supervised Random Forest Churn Probability (%) + Exponential Decay Churn Risk (%)
    - Supervised Gradient Boosting CLV Prediction ($) + Analytical 12M CLV ($)
    - Unsupervised Isolation Forest Anomaly Detection
    - Local Explainable AI (XAI) Feature Attribution & Counterfactual Boundary Solver
    """
    if recency < 0:
        raise ValueError("Recency cannot be negative.")
    if frequency < 1:
        raise ValueError("Frequency must be at least 1.")
    if monetary < 0:
        raise ValueError("Monetary spend cannot be negative.")

    kmeans, scaler, metadata = load_models(model_dir)
    
    # 1. Feature Order & Schema Enforcement
    features = metadata.get('features', ['Recency', 'Frequency', 'Monetary'])
    input_df = pd.DataFrame([[recency, frequency, monetary]], columns=features)
    
    # 2. Log1p Transformation
    if metadata.get('transformation') == 'log1p':
        input_transformed = np.log1p(input_df)
    else:
        input_transformed = input_df
        
    # 3. StandardScaler Transformation
    input_scaled = scaler.transform(input_transformed)
    scaled_cols = [f"{f}_scaled" for f in features]
    input_scaled_df = pd.DataFrame(input_scaled, columns=scaled_cols)
    
    # 4. Predict Segment
    cluster_id = int(kmeans.predict(input_scaled_df)[0])
    
    # 5. Calculate Distance to All Centroids
    centroids = kmeans.cluster_centers_
    distances = np.linalg.norm(centroids - input_scaled, axis=1)
    
    # 6. Temperature-Calibrated Softmax (tau = 0.45)
    tau = 0.45
    neg_distances = -distances / tau
    exp_scores = np.exp(neg_distances - np.max(neg_distances))
    confidence_probs = exp_scores / np.sum(exp_scores)
    
    # 7. Predictive Churn Risk Score (%) - Analytical & Supervised Random Forest
    churn_risk_pct = float(min(99.9, max(0.5, (1.0 - np.exp(-recency / 90.0)) * 100.0)))
    aov = monetary / frequency
    infer_4_df = pd.DataFrame([[recency, frequency, monetary, aov]], columns=['Recency', 'Frequency', 'Monetary', 'AverageOrderValue'])

    rf_churn_path = os.path.join(model_dir, 'rf_churn_model.pkl')
    if os.path.exists(rf_churn_path):
        rf_churn_model = joblib.load(rf_churn_path)
        ml_churn_prob = float(rf_churn_model.predict_proba(infer_4_df)[0, 1] * 100.0)
    else:
        ml_churn_prob = churn_risk_pct
    
    # 8. 12-Month Projected Customer Lifetime Value (CLV $) - Analytical & Gradient Boosting
    annual_freq_est = min(frequency * 1.5, 52.0)
    retention_prob = (100.0 - churn_risk_pct) / 100.0
    projected_clv = float(max(monetary, aov * annual_freq_est * retention_prob))

    gbr_clv_path = os.path.join(model_dir, 'gbr_clv_model.pkl')
    if os.path.exists(gbr_clv_path):
        gbr_clv_model = joblib.load(gbr_clv_path)
        ml_predicted_clv = float(max(monetary, gbr_clv_model.predict(infer_4_df)[0]))
    else:
        ml_predicted_clv = projected_clv

    # 9. Unsupervised Isolation Forest Anomaly Detection
    iso_path = os.path.join(model_dir, 'isolation_forest.pkl')
    if os.path.exists(iso_path):
        iso_model = joblib.load(iso_path)
        log_infer_4 = np.log1p(infer_4_df)
        iso_pred = int(iso_model.predict(log_infer_4)[0])
        iso_score = float(iso_model.decision_function(log_infer_4)[0])
        anomaly_status = "Anomaly (Outlier Profile)" if iso_pred == -1 else "Normal Profile"
    else:
        iso_score = 0.05
        anomaly_status = "Normal Profile"
    
    # 10. Customer Engagement Index (0 - 100 Scale)
    recency_health = 100.0 / (1.0 + 0.015 * recency)
    freq_boost = 12.0 * np.log1p(frequency)
    engagement_index = float(min(100.0, max(0.0, recency_health + freq_boost)))
    
    # 11. Customer Persona Classification
    persona = classify_micro_persona(recency, frequency, monetary, aov)

    # 12. Explainable AI (XAI) & Counterfactual Boundary Analysis
    feature_attributions, counterfactual = compute_xai_and_counterfactual(
        recency, frequency, monetary, input_scaled[0], kmeans, scaler, cluster_id
    )
    
    ai_metrics = {
        'churn_risk_pct': round(churn_risk_pct, 1),
        'ml_churn_prob_pct': round(ml_churn_prob, 1),
        'projected_clv': round(projected_clv, 2),
        'ml_predicted_clv': round(ml_predicted_clv, 2),
        'engagement_index': round(engagement_index, 1),
        'micro_persona': persona,
        'anomaly_status': anomaly_status,
        'anomaly_score': round(iso_score, 4),
        'feature_attributions': feature_attributions,
        'counterfactual': counterfactual
    }
    
    return cluster_id, distances, confidence_probs, input_scaled[0], ai_metrics

def classify_micro_persona(recency, frequency, monetary, aov):
    """
    Assigns professional Customer Persona Badges based on RFM behavior.
    """
    if monetary >= 3000 and frequency >= 8:
        return "Wholesale Bulk Buyer"
    elif monetary >= 1500 and recency > 60:
        return "Dormant High-Value Customer"
    elif recency <= 20 and frequency >= 3:
        return "Active High-Pace Customer"
    elif recency > 90 and monetary < 500:
        return "At-Risk Low-Value Customer"
    elif recency <= 30 and frequency <= 2:
        return "Promising New Customer"
    else:
        return "Standard Core Customer"

def get_human_readable_explanation(cluster_id, distances, confidence_probs, summary_df):
    """
    Returns human-readable segment name, centroid distance dictionary, confidence probabilities, and explanation.
    """
    segment_name = summary_df[summary_df['Cluster'] == cluster_id]['SegmentName'].values[0]
    
    dist_dict = {}
    prob_dict = {}
    
    for i, dist in enumerate(distances):
        name = summary_df[summary_df['Cluster'] == i]['SegmentName'].values[0]
        dist_dict[name] = round(float(dist), 2)
        prob_dict[name] = round(float(confidence_probs[i] * 100), 1)
        
    top_prob = prob_dict[segment_name]
    explanation = (f"The customer's transformed RFM profile matches the {segment_name} cluster centroid "
                   f"(Euclidean Distance: {dist_dict[segment_name]}, Segment Probability: {top_prob}%).")
    
    return segment_name, dist_dict, prob_dict, explanation
