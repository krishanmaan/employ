import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import time
import os

# Page configuration
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="👨‍💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f7fd;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 12px;
        border: none;
        padding: 10px 24px;
        transition-duration: 0.4s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .title-text {
        text-align: center;
        color: #1E88E5;
        font-size: 3em;
        font-weight: bold;
        text-shadow: 2px 2px 4px #cccccc;
        animation: fadeIn 1.5s;
    }
    .subtitle-text {
        text-align: center;
        color: #424242;
        animation: slideIn 1.5s;
    }
    @keyframes fadeIn {
        0% {opacity: 0;}
        100% {opacity: 1;}
    }
    @keyframes slideIn {
        0% {transform: translateY(30px); opacity: 0;}
        100% {transform: translateY(0); opacity: 1;}
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        animation: pulseAnimation 2s infinite;
    }
    @keyframes pulseAnimation {
        0% {box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.7);}
        70% {box-shadow: 0 0 0 10px rgba(76, 175, 80, 0);}
        100% {box-shadow: 0 0 0 0 rgba(76, 175, 80, 0);}
    }
    .card {
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        padding: 20px;
        background-color: white;
        margin-bottom: 20px;
    }
    .metric-card {
        text-align: center;
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        padding: 15px;
        background-color: white;
        transition: transform 0.3s;
    }
    .metric-card:hover {
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

# Load the pre-trained machine learning models
rf_model = joblib.load("rf_attrition_model.pkl")
gb_model = joblib.load("gb_attrition_model.pkl")
scaler = joblib.load("scaler_model.pkl")

# Page title with animation effect
st.markdown('<div class="title-text">Employee Attrition Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Predict the likelihood of employee attrition using advanced machine learning algorithms</div>', unsafe_allow_html=True)

# Create sidebar for options
st.sidebar.markdown("## Model Settings")
selected_model = st.sidebar.radio("Select Prediction Algorithm", ["Random Forest", "Gradient Boosting"])
st.sidebar.markdown("---")
show_feature_importance = st.sidebar.checkbox("Show Feature Importance", value=True)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Input Employee Details")
    
    # Input Form with two columns
    form_col1, form_col2 = st.columns(2)
    
    with form_col1:
        age = st.number_input("Age", min_value=18, max_value=65, step=1, value=30)
        income = st.number_input("Monthly Income (₹)", min_value=10000, max_value=100000, step=1000, value=30000)
        satisfaction = st.slider("Job Satisfaction (1-5)", 1, 5, 3)
        balance = st.slider("Work-Life Balance (1-4)", 1, 4, 2)
        years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, step=1, value=5)
    
    with form_col2:
        job_involvement = st.slider("Job Involvement (1-4)", 1, 4, 2)
        environment_satisfaction = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
        distance_from_home = st.number_input("Distance From Home (km)", min_value=1, max_value=30, step=1, value=10)
        performance_rating = st.slider("Performance Rating (1-5)", 1, 5, 3)
        stock_option_level = st.slider("Stock Option Level (0-3)", 0, 3, 1)
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("💡 About the Model")
    st.write("""
    Our AI model uses machine learning to predict employee attrition risk based on various factors.
    
    - **Random Forest**: An ensemble learning method that builds multiple decision trees
    - **Gradient Boosting**: A technique that builds trees sequentially to correct errors
    
    The model was trained on historical employee data with a 60-40 train-test split.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Feature importance display
if show_feature_importance:
    st.markdown("### 📈 Feature Importance")
    tab1, tab2 = st.tabs(["Random Forest", "Gradient Boosting"])
    
    with tab1:
        if os.path.exists("rf_feature_importance.png"):
            img = Image.open("rf_feature_importance.png")
            st.image(img, caption="Random Forest - Feature Importance", use_column_width=True)
        else:
            st.warning("Feature importance image not found. Please run the training script first.")
    
    with tab2:
        if os.path.exists("gb_feature_importance.png"):
            img = Image.open("gb_feature_importance.png") 
            st.image(img, caption="Gradient Boosting - Feature Importance", use_column_width=True)
        else:
            st.warning("Feature importance image not found. Please run the training script first.")

# Create a placeholder for prediction animation
prediction_placeholder = st.empty()

# Prediction button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("Predict Attrition Risk", use_container_width=True)

# Prediction logic
if predict_button:
    # Loading animation
    progress_text = "Analyzing employee data..."
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    
    # Prepare input for the model
    input_data = np.array([[
        age, income, satisfaction, balance, years_at_company, 
        job_involvement, environment_satisfaction, distance_from_home,
        performance_rating, stock_option_level
    ]])
    
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Select model and predict
    if selected_model == "Random Forest":
        prediction = rf_model.predict(input_data_scaled)
        prediction_proba = rf_model.predict_proba(input_data_scaled)[0]
    else:  # Gradient Boosting
        prediction = gb_model.predict(input_data_scaled)
        prediction_proba = gb_model.predict_proba(input_data_scaled)[0]
    
    # Calculate risk percentage
    risk_percentage = prediction_proba[1] * 100
    
    # Create visualization for risk
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_percentage,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Attrition Risk"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': risk_percentage
            }
        }
    ))
    
    fig.update_layout(height=300)
    
    # Display result with animation
    with prediction_placeholder.container():
        st.markdown('<div class="card prediction-box">', unsafe_allow_html=True)
        
        if prediction[0] == 1:
            st.markdown("### 🚨 Prediction: High Risk of Attrition")
            st.markdown(f"<h2 style='color:red; text-align:center;'>Risk Level: {risk_percentage:.1f}%</h2>", unsafe_allow_html=True)
        else:
            st.markdown("### ✅ Prediction: Low Risk of Attrition")
            st.markdown(f"<h2 style='color:green; text-align:center;'>Risk Level: {risk_percentage:.1f}%</h2>", unsafe_allow_html=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display metrics in cards
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(label="Age", value=age)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with metric_col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(label="Years at Company", value=years_at_company)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with metric_col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(label="Job Satisfaction", value=f"{satisfaction}/5")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with metric_col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(label="Work-Life Balance", value=f"{balance}/4")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Recommendations based on prediction
        st.markdown("### Recommended Actions:")
        if prediction[0] == 1:
            st.markdown("""
            - Schedule a one-on-one meeting to discuss potential concerns
            - Review compensation and benefits package
            - Explore opportunities for career advancement
            - Consider work-life balance improvements
            """)
        else:
            st.markdown("""
            - Continue regular check-ins to maintain engagement
            - Recognize and reward good performance
            - Provide growth opportunities to maintain satisfaction
            - Consider the employee for mentoring roles
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)
