import pandas as pd

# Load the dataset
df = pd.read_csv("cleaned_data.csv")

# Print the column names
print("Columns in dataset:", df.columns.tolist())

# Check if "target" is correctly defined
if "target" in df.columns:
    X = df.drop(columns=["target"])
    print("Training Features:", X.columns.tolist())
else:
    print("Error: Target column not found in dataset. Check your dataset.")
