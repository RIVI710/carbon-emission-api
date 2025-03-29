from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the saved model
model = joblib.load('carbon_emission_model.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Carbon Emission Prediction API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # Get JSON data
        features = np.array(data['features']).reshape(1, -1)  # Convert to NumPy array
        prediction = model.predict(features)  # Make prediction
        return jsonify({'prediction': prediction.tolist()})  # Return response
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)  # Run the app
