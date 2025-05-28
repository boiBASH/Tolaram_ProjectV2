# Import libraries
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder

# Load data
file_path = 'data_sample_analysis.csv'  # Change to your file location
data = pd.read_csv(file_path, encoding='ISO-8859-1')

# Preprocessing steps
data['Delivered_date'] = pd.to_datetime(data['Delivered_date'], errors='coerce', dayfirst=True)
data['Redistribution Value'] = data['Redistribution Value'].str.replace(',', '').astype(float)
data['Salesman_Code'] = data['Salesman_Code'].astype(str)

# Item-Item Collaborative Filtering
user_item_matrix = data.pivot_table(index='Customer_Phone', columns='SKU_Code', 
                                    values='Redistribution Value', aggfunc='sum', fill_value=0)
item_similarity = cosine_similarity(user_item_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity, 
                                  index=user_item_matrix.columns, 
                                  columns=user_item_matrix.columns)

# Content-Based Filtering
item_attributes = data[['SKU_Code', 'Brand', 'Branch']].drop_duplicates().set_index('SKU_Code')
encoder = OneHotEncoder()
item_features_encoded = encoder.fit_transform(item_attributes)
content_similarity = cosine_similarity(item_features_encoded)
content_similarity_df = pd.DataFrame(content_similarity, 
                                     index=item_attributes.index, 
                                     columns=item_attributes.index)

# Hybrid Similarity
common_skus = item_similarity_df.index.intersection(content_similarity_df.index)
filtered_item_similarity = item_similarity_df.loc[common_skus, common_skus]
filtered_content_similarity = content_similarity_df.loc[common_skus, common_skus]
hybrid_similarity = (filtered_item_similarity + filtered_content_similarity) / 2

# Recommendation Function
def recommend_skus_brands(customer_phone, user_item_matrix, hybrid_similarity, data, top_n=5):
    purchased_skus = user_item_matrix.loc[customer_phone]
    purchased_skus = purchased_skus[purchased_skus > 0].index.tolist()
    sku_scores = hybrid_similarity[purchased_skus].mean(axis=1)
    sku_scores = sku_scores.drop(index=purchased_skus)
    top_skus = sku_scores.sort_values(ascending=False).head(top_n)
    sku_brand_map = data[['SKU_Code', 'Brand']].drop_duplicates(subset='SKU_Code').set_index('SKU_Code')
    recommendations = sku_brand_map.loc[top_skus.index].copy()
    recommendations['Similarity_Score'] = top_skus.values
    return recommendations.reset_index()

# Combined Report Function
def combined_report(customer_phone, user_item_matrix, hybrid_similarity, data, top_n=5):
    past_purchases = data[data['Customer_Phone'] == customer_phone][['SKU_Code', 'Brand']].drop_duplicates()
    past_purchases['Type'] = 'Previously Purchased'
    recommendations = recommend_skus_brands(customer_phone, user_item_matrix, hybrid_similarity, data, top_n)
    recommendations['Type'] = 'Recommended'
    combined = pd.concat([past_purchases, recommendations[['SKU_Code', 'Brand', 'Similarity_Score', 'Type']]], ignore_index=True)
    return combined

# Example Usage
example_customer = user_item_matrix.index[0]
combined_customer_report = combined_report(example_customer, user_item_matrix, hybrid_similarity, data)

# Display in a user friendly manner
print(combined_customer_report)