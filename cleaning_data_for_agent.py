import pandas as pd
import numpy as np

# --- Data Loading & Preprocessing ---
# Load the data
df = pd.read_csv("data_sample_analysis.csv", encoding='latin1')

# Clean 'Redistribution Value'
df['Redistribution Value'] = (
    df['Redistribution Value']
    .str.replace(',', '', regex=False)
    .astype(float)
)

# Convert 'Delivered_date' to datetime
df['Delivered_date'] = pd.to_datetime(
    df['Delivered_date'], errors='coerce', dayfirst=True
)

# Fill missing 'Delivered Qty' values
df['Delivered Qty'] = df['Delivered Qty'].fillna(0)


# --- Save the cleaned DataFrame ---
# Save the cleaned version to a new CSV file
df.to_csv("data_sample_analysis_cleaned.csv", index=False)

print("Data loaded, cleaned, and saved to data_sample_analysis_cleaned.csv")