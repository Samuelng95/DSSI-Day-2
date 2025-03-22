import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="ML Model Deployment", layout="wide")

st.title('Machine Learning Model Deployment')
st.write('Predict target values by entering feature values below')

# Load diabetes dataset (example dataset)
db = datasets.load_diabetes()
df = pd.DataFrame(db.data, columns=db.feature_names)

# Select target variable
target = 'target'

# Use the diabetes target as the target column
df[target] = db.target

# Preprocessing the data
X = df.drop(columns=[target])
y = df[target]

# Handle categorical columns by getting dummies (one-hot encoding)
X = pd.get_dummies(X)

# Train/Test Split
test_size = 0.2
random_state = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=random_state)
model.fit(X_train, y_train)

# Evaluate Model (optional, for reference)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
st.write(f"Model Performance: Mean Squared Error (MSE): {mse:.2f}")

# User Input for Prediction
st.write("## Predict Target Value")
# Get the features of the dataset
feature_names = X.columns.tolist()

# Create a dictionary to store the user inputs
user_input = {}

# Get user inputs for each feature
for feature in feature_names:
    user_input[feature] = st.number_input(f"Enter value for {feature}", min_value=float(df[feature].min()), max_value=float(df[feature].max()), value=float(df[feature].mean()))

# Convert user input into a DataFrame to match the feature format
user_input_df = pd.DataFrame([user_input])

# Handle categorical columns by getting dummies (one-hot encoding)
user_input_df = pd.get_dummies(user_input_df)

# Make sure the user input has the same columns as the model expects
missing_cols = set(X.columns) - set(user_input_df.columns)
for col in missing_cols:
    user_input_df[col] = 0
user_input_df = user_input_df[X.columns]  # Reorder the columns to match X

# Make prediction using the trained model
prediction = model.predict(user_input_df)

# Display the predicted value
st.write("### Predicted Target Value:")
st.write(f"**Prediction:** {prediction[0]:.2f}")
