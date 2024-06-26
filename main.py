import pickle
import streamlit as st
import numpy as np

# Load the model
stunting_model = pickle.load(open('StuntingModel.sav', 'rb'))

# Load the scaler
scaler = pickle.load(open('scaler.sav', 'rb'))

# Title of the web app
st.title('Prediksi Stunting')

# Split columns for input fields
col1, col2 = st.columns(2)

with col1:
    Sex = st.text_input('Input Sex (1 for Male, 0 for Female)')

with col2:
    Age = st.text_input('Input Age (year)')

with col1:
    Birth_Weight = st.text_input('Input nilai Birth Weight (kg)')

with col2:
    Birth_Length = st.text_input('Input nilai Birth Length (cm)')

with col1:
    Body_Weight = st.text_input('Input nilai Body Weight (kg)')

with col2:
    Body_Length = st.text_input('Input nilai Body Length (cm)')

with col1:
    ASI_Eksklusif = st.text_input('Input ASI Eksklusif (1 for Yes, 0 for No)')

# Code for prediction
stunting_diagnosis = ''

# Function to convert inputs to numeric
def convert_to_numeric(*args):
    try:
        return [float(arg) for arg in args]
    except ValueError:
        st.error("Please enter valid numeric values.")
        return None

# Button to predict
if st.button('Test Prediksi Stunting'):
    inputs = convert_to_numeric(Sex, Age, Birth_Weight, Birth_Length, Body_Weight, Body_Length, ASI_Eksklusif)
    
    if inputs:
        # Scale the inputs
        inputs_scaled = scaler.transform([inputs])
        
        # Predict using the model
        stunting_diagnosis = stunting_model.predict(inputs_scaled)

        if stunting_diagnosis[0] == 1:
            stunting_diagnosis = 'Pasien terkena Stunting'
        else:
            stunting_diagnosis = 'Pasien tidak terkena Stunting'

        st.success(stunting_diagnosis)
