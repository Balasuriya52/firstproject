import streamlit as st
import pickle
import numpy as np

# Load the saved Linear Regression model
with open('heart_statlog_cleveland_hungary_final.sav.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


def predict_target(age, sex, resting_bp_s, cholesterol, fasting_blood_sugar, resting_ecg, max_heart_rate):
    features = np.array([age, sex, resting_bp_s, cholesterol, fasting_blood_sugar, resting_ecg, max_heart_rate])
    features = features.reshape(1, -1)
    target = model.predict(features)
    return target[0]

# Streamlit UI
st.title('Target Prediction')
st.write("""
## Input Features
Enter the values for the input features to predict EMISSION.
""")

# Input fields for user
age = st.number_input('age')
sex = st.number_input('sex')
resting_bp_s = st.number_input('resting_bp_s')
cholesterol = st.number_input('cholesterol')
fasting_blood_sugar = st.number_input('fasting_blood_sugar')
resting_ecg = st.number_input('resting_ecg')
max_heart_rate = st.number_input('max_heart_rate')

# Prediction button
if st.button('Predict'):
    # Predict target
    target_prediction = predict_target(age, sex, resting_bp_s, cholesterol, fasting_blood_sugar, resting_ecg, max_heart_rate)
    st.write(f"Prediction target: {target_prediction}")
