import pandas as pd

# Load dataset
file_path = r"C:\Users\rishi\Downloads\SupplyChainGHGEmissionFactors_v1.2_NAICS_CO2e_USD2021.csv"
df = pd.read_csv(file_path)

# Display first few rows
print(df.head())

# Check dataset information
print(df.info())

# Check for missing values
print(df.isnull().sum())
