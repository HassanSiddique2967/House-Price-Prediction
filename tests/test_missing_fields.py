import pytest
import requests

BASE_URL = 'http://localhost:5000/predict'  # Adjust if running on a different port

# Example data with missing fields
missing_field_data = {
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

def test_missing_fields():
    response = requests.post(BASE_URL, json=missing_field_data)
    assert response.status_code == 400
    json_response = response.json()
    assert 'error' in json_response
    assert 'Missing field' in json_response['error']

if __name__ == "__main__":
    pytest.main()
