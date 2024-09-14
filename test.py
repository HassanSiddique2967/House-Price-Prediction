import unittest
import joblib
import numpy as np
import pandas as pd

class TestModel(unittest.TestCase):

    def setUp(self):
        # Load the saved model and scaler
        self.model = joblib.load('house_price_rf_model.pkl')
        self.scaler = joblib.load('scaler.pkl')
        
        # Define a sample input (after preprocessing)
        self.sample_data = np.array([[3, 2, 2000, 5000, 1, 3, 1500, 500, 0, 0, 1985, 0]])  # Example input
    
    def test_model_prediction(self):
        # Scale the input
        sample_scaled = self.scaler.transform(self.sample_data)
        
        # Predict using the loaded model
        prediction = self.model.predict(sample_scaled)
        print(prediction)
        # Check that the prediction is a valid number
        self.assertTrue(prediction[0] > 0, "Prediction should be a positive value")
    
    def test_model_file_exists(self):
        # Check if model file exists
        try:
            joblib.load('house_price_rf_model.pkl')
        except FileNotFoundError:
            self.fail('Model file not found')

if __name__ == '__main__':
    unittest.main()
 
