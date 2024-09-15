from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('house_price_rf_model.pkl')
scaler = joblib.load('scaler.pkl')

# Serve the index.html file on the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Check if the data is None or improperly formatted
    if data is None:
        return jsonify({'error': 'Invalid JSON data'}), 400

    # List of required fields
    required_fields = [
        'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 
        'floors', 'condition', 'sqft_above', 'sqft_basement', 
        'waterfront', 'view', 'yr_built', 'yr_renovated'
    ]

    # Validate that all required fields are present
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing field: {field}'}), 400

    # Prepare the feature array
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

    # Log feature shape for debugging
    print("Features shape:", features.shape)

    # Scale the features
    try:
        features_scaled = scaler.transform(features)
    except Exception as e:
        return jsonify({'error': f'Error scaling features: {str(e)}'}), 500

    # Make the prediction
    try:
        prediction = model.predict(features_scaled)
    except Exception as e:
        return jsonify({'error': f'Error during prediction: {str(e)}'}), 500

    # Return the prediction as a JSON response
    return jsonify({'predicted_price': prediction[0]})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
