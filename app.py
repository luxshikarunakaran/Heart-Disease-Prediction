import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.write("""
# Heart Disease Prediction App

This app predicts if a patient has heart disease.

Data obtained from [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset).
""")

st.sidebar.header('User Input Features')

# Collects user input features into dataframe
def user_input_features():
    age = st.sidebar.number_input('Enter your age: ', min_value=0, max_value=120, step=1)
    sex = st.sidebar.selectbox('Sex', (0, 1))
    cp = st.sidebar.selectbox('Chest pain type', (0, 1, 2, 3))
    trestbps = st.sidebar.number_input('Resting blood pressure: ', min_value=0, max_value=300, step=1)
    chol = st.sidebar.number_input('Serum cholesterol in mg/dl: ', min_value=0, max_value=1000, step=1)
    fbs = st.sidebar.selectbox('Fasting blood sugar > 120 mg/dl', (0, 1))
    restecg = st.sidebar.selectbox('Resting electrocardiographic results', (0, 1, 2))
    thalach = st.sidebar.number_input('Maximum heart rate achieved: ', min_value=0, max_value=250, step=1)
    exang = st.sidebar.selectbox('Exercise induced angina', (0, 1))
    oldpeak = st.sidebar.number_input('Oldpeak: ', min_value=0.0, max_value=10.0, step=0.1)
    slope = st.sidebar.selectbox('The slope of the peak exercise ST segment', (0, 1, 2))
    ca = st.sidebar.selectbox('Number of major vessels (0-3) colored by flourosopy', (0, 1, 2, 3, 4))
    thal = st.sidebar.selectbox('Thal', (0, 1, 2, 3))

    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Read in the heart disease dataset for encoding consistency
heart_dataset = pd.read_csv('heart.csv')
heart_dataset = heart_dataset.drop(columns=['target'])

df = pd.concat([input_df, heart_dataset], axis=0)

# Encoding of categorical features
df = pd.get_dummies(df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

# Ensure the order of columns matches the model's training set
df = df[:1]  # Select only the first row (the user input data)

# Load the model
load_clf = pickle.load(open('Random_forest_model.pkl', 'rb'))

# Get model expected feature names
model_features = load_clf.feature_names_in_

# Check for missing columns and add them if necessary
missing_features = set(model_features) - set(df.columns)
for feature in missing_features:
    df[feature] = 0

# Ensure the DataFrame columns are in the correct order
df = df[model_features]

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

st.subheader('Prediction')
st.write('Heart Disease' if prediction[0] == 1 else 'No Heart Disease')

st.subheader('Prediction Probability')
st.write(prediction_proba)
