# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import joblib

file_path = './data/data.csv'  # Ensure the correct path to your data
home_data = pd.read_csv(file_path)

# Define the target variable (house prices)
y = home_data['price']

# Define a more comprehensive set of features
features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'condition', 
            'sqft_above', 'sqft_basement', 'waterfront', 'view', 'yr_built', 'yr_renovated']
X = home_data[features]

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.9, test_size=0.1, random_state=0)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

# Initialize the Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=0)

# Fit the model to the training data
rf_model.fit(X_train_scaled, y_train)

# Make predictions on the validation data
y_pred_rf = rf_model.predict(X_valid_scaled)

# Calculate Mean Absolute Error (MAE)
mae_rf = mean_absolute_error(y_valid, y_pred_rf)
print(f"Random Forest Mean Absolute Error: {mae_rf}")

# Optional: Display a few predictions alongside actual values for comparison
comparison = pd.DataFrame({'Actual Price': y_valid, 'Predicted Price': y_pred_rf})
print(comparison.head())

# Save the trained model
joblib.dump(rf_model, 'house_price_rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')  # Save the scaler for future use
 
