import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned data
df = pd.read_csv("cleaned_data.csv")

# Display first few rows
print(df.head())

# Check the structure of the data
print("\nData Info:")
print(df.info())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Select only numerical columns for correlation
num_cols = df.select_dtypes(include=['number'])  # Only select numeric columns

# Check correlations again
print("\nCorrelation Matrix:")
print(num_cols.corr())

# Visualize correlations using a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(num_cols.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()
