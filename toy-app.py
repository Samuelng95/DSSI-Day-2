import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Set page config
apptitle = 'DSSI Toy App'
st.set_page_config(page_title=apptitle, layout='wide')

st.title('Diabetes Classification')
st.write('A simple model to classify if someone is diabetic or not.')

# Load diabetes dataset
db = datasets.load_diabetes()

# Create a DataFrame
df = pd.DataFrame(db.data, columns=db.feature_names)
target = db.target

# Convert the regression target into a binary classification target
binary_target = (target > 150).astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df, binary_target, test_size=0.2, random_state=42)

# Standardize the data (important for models like Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Logistic Regression classifier
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Display results
st.subheader('Model Accuracy')
st.write(f"Accuracy: {accuracy:.2f}")

st.subheader('Classification Report')
st.text(report)

# --- New Section: User Input ---
st.subheader('Enter Your Data for Prediction')

# Create number input widgets for the user to enter their data
age = st.number_input('Age', min_value=0, max_value=120, value=50)
sex = st.number_input('Sex (0 = Female, 1 = Male)', min_value=0, max_value=1, value=1)
bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, value=25.0)
bp = st.number_input('Blood Pressure (mm Hg)', min_value=50, max_value=200, value=80)
s1 = st.number_input('S1', min_value=0.0, max_value=10.0, value=0.0)
s2 = st.number_input('S2', min_value=0.0, max_value=10.0, value=0.0)
s3 = st.number_input('S3', min_value=0.0, max_value=10.0, value=0.0)
s4 = st.number_input('S4', min_value=0.0, max_value=10.0, value=0.0)
s5 = st.number_input('S5', min_value=0.0, max_value=10.0, value=0.0)
s6 = st.number_input('S6', min_value=0.0, max_value=10.0, value=0.0)

# Store the input data in a DataFrame
user_data = np.array([[age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]])
user_data_scaled = scaler.transform(user_data)

# Predict using the trained model
prediction = model.predict(user_data_scaled)

# Show the prediction result
if prediction == 1:
    st.write("The model predicts that the person is **diabetic**.")
else:
    st.write("The model predicts that the person is **not diabetic**.")
