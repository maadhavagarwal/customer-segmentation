import unittest
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from preprocessing import clean_data
from feature_engineering import compute_rfm_features, compute_additional_features
from clustering import preprocess_for_clustering
from inference import predict_new_customer, classify_micro_persona
from analytics import analyze_temporal_sales
from recommendations import get_personalized_recommendation_strategy

class TestPipeline(unittest.TestCase):
    def setUp(self):
        data = {
            'InvoiceNo': ['123', '123', 'C125', '126', '127'],
            'StockCode': ['A', 'B', 'C', 'A', 'D'],
            'Description': ['Item A', 'Item B', 'Item C', 'Item A', 'Item D'],
            'Quantity': [10, 5, -5, 0, 20],
            'InvoiceDate': pd.to_datetime(['2023-01-01 10:00:00', '2023-01-01 10:00:00', '2023-01-03 14:00:00', '2023-01-04 15:00:00', '2023-01-05 11:00:00']),
            'UnitPrice': [1.5, 2.0, 3.0, 1.5, 5.0],
            'CustomerID': [1001, 1001, 1002, None, 1003],
            'Country': ['UK', 'UK', 'UK', 'UK', 'UK']
        }
        self.df = pd.DataFrame(data)
        
    def test_preprocessing(self):
        clean_df = clean_data(self.df)
        self.assertFalse(clean_df['CustomerID'].isnull().any())
        self.assertFalse(clean_df['InvoiceNo'].astype(str).str.startswith('C').any())
        self.assertTrue((clean_df['Quantity'] > 0).all())
        self.assertTrue((clean_df['UnitPrice'] > 0).all())
        self.assertEqual(len(clean_df), 3)
        
    def test_rfm_and_ai_features(self):
        clean_df = clean_data(self.df)
        rfm = compute_rfm_features(clean_df)
        full_df = compute_additional_features(clean_df, rfm)
        
        self.assertTrue('Recency' in full_df.columns)
        self.assertTrue('Frequency' in full_df.columns)
        self.assertTrue('Monetary' in full_df.columns)
        self.assertTrue('ChurnRiskPct' in full_df.columns)
        self.assertTrue('ProjectedCLV_12M' in full_df.columns)
        self.assertTrue('EngagementIndex' in full_df.columns)
        
        self.assertTrue((full_df['ChurnRiskPct'] >= 0.5).all())
        self.assertTrue((full_df['ProjectedCLV_12M'] > 0).all())

    def test_micro_persona(self):
        persona = classify_micro_persona(recency=5, frequency=10, monetary=4500.0, aov=450.0)
        self.assertEqual(persona, "👑 Wholesale Power VIP")

    def test_high_confidence_inference(self):
        model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
        if os.path.exists(os.path.join(model_dir, 'kmeans_model.pkl')):
            c_id, dists, probs, scaled, ai_metrics = predict_new_customer(5, 15, 4500.0, model_dir=model_dir)
            top_prob = probs[c_id] * 100.0
            self.assertGreaterEqual(top_prob, 80.0) # High confidence calibrated check
            self.assertIn('churn_risk_pct', ai_metrics)
            self.assertIn('projected_clv', ai_metrics)

if __name__ == '__main__':
    unittest.main()
