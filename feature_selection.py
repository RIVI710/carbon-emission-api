import pandas as pd

# Load the cleaned data
df = pd.read_csv("cleaned_data.csv")

# Selecting numerical columns only
num_cols = df.select_dtypes(include=['number'])

# Drop irrelevant columns (modify based on your dataset)
columns_to_drop = ["2017 NAICS Code"]  # Drop if not useful
df_selected = num_cols.drop(columns=columns_to_drop, errors='ignore')

# Save the selected features
df_selected.to_csv("selected_features.csv", index=False)

print("✅ Feature selection complete! Saved as 'selected_features.csv'.")
print(df_selected.head())
