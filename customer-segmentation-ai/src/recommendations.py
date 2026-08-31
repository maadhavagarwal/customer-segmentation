import pandas as pd
import json

def generate_segment_recommendations(clean_df, rfm_df, top_n=5):
    """
    Computes top products favored by customers in each segment cluster.
    """
    print("Generating segment-based product recommendations...")
    merged = pd.merge(clean_df, rfm_df[['CustomerID', 'SegmentName', 'Cluster']], on='CustomerID')
    merged['Description'] = merged['Description'].astype(str).str.strip().str.upper()
    
    segment_recommendations = {}
    
    for segment_name in merged['SegmentName'].unique():
        seg_data = merged[merged['SegmentName'] == segment_name]
        
        top_products = seg_data.groupby('Description').agg({
            'Quantity': 'sum',
            'TotalAmount': 'sum',
            'CustomerID': 'nunique'
        }).reset_index()
        
        top_products.rename(columns={
            'Quantity': 'TotalQuantity',
            'TotalAmount': 'TotalRevenue',
            'CustomerID': 'BuyersCount'
        }, inplace=True)
        
        top_products = top_products.sort_values(by='TotalRevenue', ascending=False).head(top_n)
        segment_recommendations[segment_name] = top_products.to_dict(orient='records')
        
    print("Segment recommendations generated.")
    return segment_recommendations

def get_item_recommendations(selected_item, market_basket_df, top_n=5):
    """
    Given a selected product description, finds top co-purchased items using market basket rules.
    """
    if market_basket_df.empty or not selected_item:
        return []
        
    selected_clean = selected_item.strip().upper()
    
    # Match Item_A or Item_B
    rules_a = market_basket_df[market_basket_df['Item_A'] == selected_clean].copy()
    rules_a.rename(columns={'Item_B': 'RecommendedProduct', 'Confidence_A_to_B': 'Confidence'}, inplace=True)
    
    rules_b = market_basket_df[market_basket_df['Item_B'] == selected_clean].copy()
    rules_b.rename(columns={'Item_A': 'RecommendedProduct', 'Confidence_B_to_A': 'Confidence'}, inplace=True)
    
    combined = pd.concat([rules_a[['RecommendedProduct', 'Confidence', 'Lift', 'Support']], 
                          rules_b[['RecommendedProduct', 'Confidence', 'Lift', 'Support']]])
    
    if combined.empty:
        return []
        
    combined = combined.sort_values(by=['Lift', 'Confidence'], ascending=False).head(top_n)
    return combined.to_dict(orient='records')

def get_personalized_recommendation_strategy(segment_name):
    """
    Returns marketing strategies and next-best-action guidance for a segment.
    """
    strategies = {
        "Champions": {
            "title": "VIP & Ambassador Retention",
            "action": "Reward loyalty with VIP exclusive perks, early product access, and referral bonuses.",
            "channel": "Direct SMS / Personal Concierge / Exclusive Email",
            "discount_strategy": "No aggressive discounts needed. Offer exclusive value-add items & early access."
        },
        "Loyal Customers": {
            "title": "Up-Sell & Relationship Building",
            "action": "Offer subscription models, loyalty points booster, and cross-sell higher value product tiers.",
            "channel": "Personalized Email / In-App Notifications",
            "discount_strategy": "Tiered rewards (e.g. Spend $100 get $15 voucher)."
        },
        "At-Risk Customers": {
            "title": "Re-Engagement & Win-Back",
            "action": "Trigger win-back campaign featuring items they previously purchased or complementary bestsellers.",
            "channel": "Targeted Retargeting Ads / Urgency-based Email",
            "discount_strategy": "Limited-time 15-20% discount coupon to incentivize immediate purchase."
        },
        "Lost Customers": {
            "title": "Reactivation or Low-Cost Nurturing",
            "action": "Automated re-engagement drip campaign showcasing brand new arrivals and heavy clearance items.",
            "channel": "Low-cost Email Marketing / Social Retargeting",
            "discount_strategy": "Deep clearance incentives or free shipping on next purchase."
        }
    }
    
    default_strategy = {
        "title": "Standard Engagement",
        "action": "Send general product updates and popular bestsellers.",
        "channel": "Email Newsletter",
        "discount_strategy": "Standard promotional discounts."
    }
    
    return strategies.get(segment_name, default_strategy)

if __name__ == "__main__":
    pass
