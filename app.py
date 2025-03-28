import streamlit as st
import joblib
import numpy as np

# Load the pre-trained machine learning model
model = joblib.load("employee_attrition_model.pkl")  # Replace with your model filename

# Page title
st.title("Employee Attrition Prediction")
st.write("Predict the likelihood of employee attrition based on various factors.")

# Input Form
st.header("Input Employee Details")
age = st.number_input("Enter Age", min_value=18, max_value=65, step=1)
income = st.number_input("Enter Monthly Income", min_value=1000, max_value=100000, step=500)
satisfaction = st.slider("Job Satisfaction (1-5)", 1, 5)
balance = st.slider("Work-Life Balance (1-4)", 1, 4)
years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, step=1)
job_involvement = st.slider("Job Involvement (1-4)", 1, 4)
environment_satisfaction = st.slider("Environment Satisfaction (1-4)", 1, 4)

# Prediction button
if st.button("Predict"):
    # Prepare input for the model
    input_data = np.array([[age, income, satisfaction, balance, years_at_company, job_involvement, environment_satisfaction]])
    prediction = model.predict(input_data)

    # Display result
    if prediction[0] == 1:
        st.error("Prediction: High Risk of Attrition")
    else:
        st.success("Prediction: Low Risk of Attrition")
