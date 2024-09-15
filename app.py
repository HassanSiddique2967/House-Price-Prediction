
from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('house_price_rf_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return "House Price Prediction API"

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json()

    # Extract the features from the input data
    features = np.array([[
        data['bedrooms'],
        data['bathrooms'],
        data['sqft_living'],
        data['sqft_lot'],
        data['floors'],
        data['condition'],
        data['sqft_above'],
        data['sqft_basement'],
        data['waterfront'],
        data['view'],
        data['yr_built'],
        data['yr_renovated']
    ]])

    # Scale the features
    features_scaled = scaler.transform(features)

    # Make the prediction using the loaded model
    prediction = model.predict(features_scaled)

    # Return the prediction as a JSON response
    return jsonify({'predicted_price': prediction[0]})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

