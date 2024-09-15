import requests
import pytest

BASE_URL = 'http://localhost:5000/predict'  # Adjust if running on a different port

# Example valid data for testing
valid_data = {
    "bedrooms": 3,
    "bathrooms": 1.5,
    "sqft_living": 1340,
    "sqft_lot": 7912,
    "floors": 1.5,
    "condition": 3,
    "sqft_above": 1340,
    "sqft_basement": 0,
    "waterfront": 0,
    "view": 0,
    "yr_built": 1955,
    "yr_renovated": 2005
}

def test_scaling_errors(monkeypatch):
    def mock_transform(*args, **kwargs):
        raise Exception("Scaling error")

    monkeypatch.setattr('joblib.load', lambda x: mock_transform)
    
    response = requests.post(BASE_URL, json=valid_data)
    assert response.status_code == 500
    json_response = response.json()
    assert 'error' in json_response
    assert 'Error scaling features' in json_response['error']

if __name__ == "__main__":
    pytest.main()
