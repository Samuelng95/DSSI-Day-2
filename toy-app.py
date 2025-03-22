import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
apptitle = 'DSSI Toy App'

st.set_page_config(page_title=apptitle, layout='wide')

st.title('My First Streamlit Application')
st.write('Reference: https://docs.streamlit.io/en/stable/api.html#display-data')
st.balloons() 

# Load diabetes dataset
st.subheader('**Diabetes Data**')
db = datasets.load_diabetes()

df = pd.DataFrame(db.data, columns=db.feature_names)

col1, col2 = st.columns([2,1])
with col1:
    # Display dataframe as an interactive table
    st.dataframe(df, use_container_width=True)
with col2:
    # Plot histogram for age of patients
    fig, ax = plt.subplots(figsize=(6, 3))
    if 1==0: # Evaluate True to show plot
        df['age'].hist(bins = 10, ax=ax)
        fig.suptitle("Age Distribution")
        st.pyplot(fig)
# Create a figure and axis
fig, ax = plt.subplots(figsize=(6, 3))

# Plot histogram
df['age'].hist(bins=10, ax=ax)

# Set title
fig.suptitle("Age Distribution")

# Display the plot in Streamlit
st.pyplot(fig)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

# Set Streamlit page config
st.set_page_config(page_title="ML Model Deployment", layout="wide")

# Title
st.title("📊 General ML Model Deployment")

# Upload dataset
st.sidebar.header("Upload Your Dataset 📂")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    # Read data
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of the Dataset")
    st.dataframe(df.head())

    # Select target variable
    target = st.sidebar.selectbox("Select Target Column", df.columns)

    # Choose ML model type
    model_type = st.sidebar.radio("Choose Model Type", ["Classification", "Regression"])

    # Train/Test Split
    st.sidebar.write("### Train/Test Split")
    test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20, step=5) / 100
    random_state = st.sidebar.number_input("Random Seed", value=42)

    # Splitting Data
    X = df.drop(columns=[target])
    y = df[target]

    # Convert categorical features if needed
    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Initialize Model
    if model_type == "Classification":
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=random_state)

    # Train Model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluate Model
    st.write("## Model Performance")
    if model_type == "Classification":
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"**Accuracy:** {accuracy:.2f}")
    else:
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"**Mean Squared Error:** {mse:.2f}")

    # Feature Importance
    st.write("## Feature Importance")
    feature_importance = pd.Series(model.feature_importances_, index=X.columns)
    fig, ax = plt.subplots(figsize=(8, 4))
    feature_importance.nlargest(10).plot(kind='barh', ax=ax)
    ax.set_title("Top 10 Important Features")
    st.pyplot(fig)

    # Correlation Heatmap
    st.write("## Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    st.pyplot(fig)


