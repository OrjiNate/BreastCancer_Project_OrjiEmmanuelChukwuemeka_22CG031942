import streamlit as st
import pandas as pd
import joblib
import os

# Robust path handling
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'model', 'breast_cancer_model.pkl')

# Load the saved model
model = joblib.load(model_path)

st.set_page_config(page_title="Breast Cancer Predictor", page_icon="🎗️")
st.title("🎗️ Breast Cancer Prediction System")

st.info("Educational Purpose Only: Not a medical diagnostic tool.")

# User Inputs
st.subheader("Input Tumor Features")
radius = st.number_input("Mean Radius", value=14.0)
texture = st.number_input("Mean Texture", value=19.0)
perimeter = st.number_input("Mean Perimeter", value=92.0)
area = st.number_input("Mean Area", value=650.0)
smoothness = st.number_input("Mean Smoothness", value=0.1)

if st.button("Predict Diagnosis"):
    # Must match the training feature names exactly
    input_df = pd.DataFrame([[radius, texture, perimeter, area, smoothness]], 
                            columns=['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness'])
    
    prediction = model.predict(input_df)[0]
    
    # In sklearn dataset: 0 is Malignant, 1 is Benign
    if prediction == 0:
        st.error("### Prediction: Malignant")
    else:
        st.success("### Prediction: Benign")