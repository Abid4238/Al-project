# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 16:55:57 2023

@author: ishti
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the pre-trained model
loaded_model = joblib.load('E:/Machine Learning Project/Diabatics data/model_for_diabetics.pkl', 'rb')

# Define the function to make predictions
def predict_diabetes(input_data):
    input_df = pd.DataFrame(input_data, columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    prediction = loaded_model.predict(input_df)
    return prediction[0]

# Streamlit App
st.title('Diabetes Prediction App')

# Input form
st.sidebar.header('Input Features:')
pregnancies = st.sidebar.number_input('Pregnancies', min_value=0, max_value=20, value=0)
glucose = st.sidebar.number_input('Glucose', min_value=0, max_value=200, value=0)
# Add other input features...

input_data = [pregnancies, glucose]  # Add other input features...

# Make prediction
if st.sidebar.button('Predict'):
    result = predict_diabetes(input_data)
    if result == 0:
        st.success('The person is not diabetic.')
    else:
        st.error('The person is diabetic.')

