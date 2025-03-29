from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# Load data
df = pd.read_csv("selected_features.csv")

# Define target variable
target_column = "Supply Chain Emission Factors without Margins"
X = df.drop(columns=[target_column], errors='ignore')
y = df[target_column]

# Apply Standard Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save the model
import joblib
joblib.dump(model, "trained_model.pkl")
