import numpy as np
import pandas as pd
import joblib
import json
import os

def load_models(model_dir='../models'):
    """
    Loads trained K-Means model, scaler, and metadata.
    """
    kmeans = joblib.load(os.path.join(model_dir, 'kmeans_model.pkl'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    with open(os.path.join(model_dir, 'model_metadata.json'), 'r') as f:
        metadata = json.load(f)
    return kmeans, scaler, metadata

def predict_new_customer(recency, frequency, monetary, model_dir='../models'):
    """
    Inference Engine.
    - Temperature-Calibrated Softmax (tau = 0.45) for Segment Probability
    - Churn Risk Probability (%)
    - 12-Month Projected Customer Lifetime Value (CLV $)
    - Customer Engagement Index (0 - 100)
    - Customer Persona Classification
    """
    # Input validation
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
    
    # 7. Predictive Churn Risk Score (%)
    churn_risk_pct = float(min(99.9, max(0.5, (1.0 - np.exp(-recency / 90.0)) * 100.0)))
    
    # 8. 12-Month Projected Customer Lifetime Value (CLV $)
    aov = monetary / frequency
    annual_freq_est = min(frequency * 1.5, 52.0)
    retention_prob = (100.0 - churn_risk_pct) / 100.0
    projected_clv = float(max(monetary, aov * annual_freq_est * retention_prob))
    
    # 9. Customer Engagement Index (0 - 100 Scale)
    recency_health = 100.0 / (1.0 + 0.015 * recency)
    freq_boost = 12.0 * np.log1p(frequency)
    engagement_index = float(min(100.0, max(0.0, recency_health + freq_boost)))
    
    # 10. Customer Persona Classification
    persona = classify_micro_persona(recency, frequency, monetary, aov)
    
    ai_metrics = {
        'churn_risk_pct': round(churn_risk_pct, 1),
        'projected_clv': round(projected_clv, 2),
        'engagement_index': round(engagement_index, 1),
        'micro_persona': persona
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
