import pandas as pd

# Load the dataset (use the same path from data_cleaning.py)
file_path = r"C:\Users\rishi\Downloads\SupplyChainGHGEmissionFactors_v1.2_NAICS_CO2e_USD2021.csv"
df = pd.read_csv(file_path)

# Check basic information
print("Basic Info:")
print(df.info())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check basic statistics
print("\nDescriptive Statistics:")
print(df.describe())

# Display column names
print("\nColumn Names:")
print(df.columns)
