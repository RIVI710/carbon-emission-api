import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# ✅ Correct way to load the model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = [data['features']]

    # Ensure model has `predict` method before calling it
    if hasattr(model, 'predict'):
        prediction = model.predict(features)
        return jsonify({'prediction': prediction.tolist()})
    else:
        return jsonify({'error': 'Model is not correctly loaded.'})


if __name__ == '__main__':
    app.run(debug=True)
