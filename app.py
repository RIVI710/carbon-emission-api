from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return "Carbon Emission API is Live!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)  # Ensure the correct port is used
