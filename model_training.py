import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load selected features
df = pd.read_csv("selected_features.csv")

# Define target variable (modify if needed)
target_column = "Supply Chain Emission Factors without Margins"
X = df.drop(columns=[target_column], errors='ignore')
y = df[target_column]

# Split into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Model Training Complete!")
print(f"📊 Mean Absolute Error: {mae}")
print(f"📈 R-squared Score: {r2}")



