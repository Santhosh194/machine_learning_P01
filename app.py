from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

# Initialize app
app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Label mapping (Iris dataset)
labels = ["setosa", "versicolor", "virginica"]

# Home route
@app.route("/")
def home():
    return "✅ ML Model API is running!"

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Validate input
        if not data or "features" not in data:
            return jsonify({"error": "Please provide 'features' in request"}), 400

        features = data["features"]

        # Validate length (Iris expects 4 features)
        if len(features) != 4:
            return jsonify({"error": "Exactly 4 features required"}), 400

        # Convert to numpy array
        features = np.array(features).reshape(1, -1)

        # Scale input
        features = scaler.transform(features)

        # Predict
        prediction = model.predict(features)[0]

        # Convert to label
        prediction_label = labels[prediction]

        return jsonify({
            "prediction": prediction_label
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run app (important for Render)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
