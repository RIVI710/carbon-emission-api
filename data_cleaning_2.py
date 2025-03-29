import pandas as pd

# Load the dataset
file_path = r"C:\Users\rishi\Downloads\SupplyChainGHGEmissionFactors_v1.2_NAICS_CO2e_USD2021.csv"
df = pd.read_csv(file_path)

# 1️⃣ Drop duplicate rows (if any)
df = df.drop_duplicates()

# 2️⃣ Handle missing values
df = df.dropna()  # Removes rows with missing values
# OR
# df.fillna(value=0, inplace=True)  # Replace missing values with 0

# 3️⃣ Convert data types if needed
# Example: Convert a column to float (change 'column_name' to actual column name)
# df['column_name'] = df['column_name'].astype(float)

# 4️⃣ Rename columns (if necessary)
# df.rename(columns={"OldColumnName": "NewColumnName"}, inplace=True)

# 5️⃣ Save the cleaned dataset
df.to_csv("cleaned_data.csv", index=False)

print("✅ Data cleaning completed! Cleaned file saved as 'cleaned_data.csv'.")
