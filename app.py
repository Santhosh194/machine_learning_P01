from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route('/')
def home():
    return "ML Model API is running"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Input features
        features = np.array(data['features']).reshape(1, -1)
        
        # Scale input
        features = scaler.transform(features)
        
        # Predict
        prediction = model.predict(features)[0]
        
        return jsonify({
            "prediction": int(prediction)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
