import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the models
lin_model = joblib.load('insurance_linear.sav')
log_model = joblib.load('insurance_logistic.sav')
model_columns = joblib.load('model_columns.sav')

st.title('Medical Cost & High-Risk Prediction Tool')
st.write('Enter the patient details below to predict medical charges or high-cost risk.')

# Sidebar for prediction type
prediction_type = st.sidebar.radio(
    "What would you like to predict?",
    ("Medical Charges (Linear)", "High-Cost Risk (Logistic)")
)

# Input fields
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)

with col2:
    sex_input = st.selectbox("Sex", ['male', 'female'])
    region_input = st.selectbox("Region", ['northeast', 'northwest', 'southeast', 'southwest'])
    smoker_input = st.selectbox("Smoker", ['no', 'yes'])

if st.button('Predict', type="primary"):
    # --- MANUAL PREPROCESSING (Must match Colab exactly) ---
    sex = 0 if sex_input == 'male' else 1
    smoker = 1 if smoker_input == 'yes' else 0
    region_map = {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}
    region = region_map[region_input]
    
    # Feature Engineering
    smoker_bmi_interaction = smoker * bmi

    # Create the input array in the exact order of model_columns
    input_data = {
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region],
        'smoker_bmi_interaction': [smoker_bmi_interaction]
    }
    input_df = pd.DataFrame(input_data)
    input_df = input_df[model_columns] # Ensure column order

    if prediction_type == "Medical Charges (Linear)":
        prediction = lin_model.predict(input_df)[0]
        st.success(f'Predicted Medical Charges: **${prediction:,.2f}**')
    else:
        proba = log_model.predict_proba(input_df)[0][1]
        prediction = log_model.predict(input_df)[0]
        
        status = "🔴 HIGH RISK" if prediction == 1 else "🟢 LOW RISK"
        st.success(f"### High-Cost Risk: {status}")
        st.info(f"Probability of being a high-cost patient: {proba:.2%}")
