import pickle

try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
    print("Model type:", type(model))
    # Check if the loaded model has a predict method
    if hasattr(model, "predict"):
        print("The model has a 'predict' method.")
    else:
        print("The model does NOT have a 'predict' method.")
except Exception as e:
    print("Error loading model:", e)
