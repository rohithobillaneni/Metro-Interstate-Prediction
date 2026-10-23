import streamlit as st
import requests
import os
import math

from src.offline_data import hour_category as derive_hour_category
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Metro Traffic Prediction",
    page_icon="🚦",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
        /* General Page Styling */
        body { background-color: #f4f4f4; font-family: 'Arial', sans-serif; }
        .main-title { text-align: center; color: #2c3e50; font-size: 36px; font-weight: bold; margin-bottom: 10px; }
        .sub-title { text-align: center; color: #7f8c8d; font-size: 20px; }
        
        /* Sidebar Styling */
        .sidebar .sidebar-content { background-color: #ecf0f1; padding: 20px; border-radius: 10px; }

        /* Button Styling */
        .stButton>button { 
            background-color: #3498db; color: white; border-radius: 10px; 
            padding: 12px 20px; font-size: 16px; transition: 0.3s;
        }
        .stButton>button:hover { background-color: #2980b9; }

        /* Footer */
        .footer { text-align: center; font-size: 14px; color: #7f8c8d; margin-top: 20px; }
    </style>
""", unsafe_allow_html=True)

# Title and Description
st.markdown("<h1 class='main-title'>🚦 Metro Traffic Volume Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Estimate same-hour traffic volume using calendar and observed weather.</p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar for Inputs
st.sidebar.header("📝 Enter Traffic Conditions")

# User Inputs
temp = st.sidebar.slider("🌡 Temperature (°C)", -10, 40, 15)
clouds_all = st.sidebar.slider("☁ Cloud Coverage (%)", 0, 100, 50)
holiday = st.sidebar.selectbox("🎉 Is it a holiday?", ["No", "Yes"])
weather_main = st.sidebar.selectbox("🌦 Weather Condition", ["Clear", "Clouds", "Rain", "Snow", "Fog"])
weekday = st.sidebar.selectbox("📆 Day of the Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
hour = st.sidebar.number_input("⏰ Hour (0-23)", min_value=0, max_value=23, value=12)
hour_category = derive_hour_category(hour)
st.sidebar.caption(f"Time of day: {hour_category}")
month = st.sidebar.selectbox("📅 Month", ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])

# Predict Button (Centered)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🚀 Predict Traffic Volume")

# Prediction Logic
if predict_button:
    with st.spinner("🔄 Predicting..."):

        
        #If running locally use below url (api_url = "http://127.0.0.1:8000/predict") and 
        #if deploying using dockers please use below url (api_url = "http://traffic-backend:8000/predict)


        # API request
        #api_url = "http://127.0.0.1:8000/predict"
        api_url = os.environ.get("METRO_API_URL", "http://127.0.0.1:8000/predict")
        data = {
            "holiday": holiday,
            "temp": temp,
            "clouds_all": clouds_all,
            "weather_main": weather_main,
            "weekday": weekday,
            "hour": hour,
            "hour_category": hour_category,
            "month": month
        }
        try:
            response = requests.post(api_url, json=data, timeout=10)
        except requests.RequestException:
            st.error("The prediction API could not be reached. Check its URL and startup status.")
            st.stop()

        if response.status_code != 200:
            st.error("❌ API request failed. Ensure FastAPI is running.")
            st.stop()
        try:
            result = response.json()
        except requests.exceptions.JSONDecodeError:
            st.error("The prediction API returned invalid JSON.")
            st.stop()
        prediction = result.get("prediction") if isinstance(result, dict) else None
        try:
            valid_prediction = (
                type(prediction) in (int, float)
                and math.isfinite(prediction)
                and prediction >= 0
            )
        except OverflowError:
            valid_prediction = False
        if not valid_prediction:
            st.error("⚠️ Invalid prediction received. Expected a finite, nonnegative number.")
            st.stop()

        st.markdown("### 🚗 **Estimated Hourly Traffic Volume**")
        st.success(f"🔹 **{prediction:,.0f} vehicles/hour**")

        # Volume does not establish congestion thresholds, so use a neutral scale.
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            number={'valueformat': ',.0f'},
            title={'text': "Vehicles/hour"},
            gauge={'axis': {'range': [0, max(10000, prediction)]},
                   'bar': {'color': "#3498db"}}
        ))
        st.plotly_chart(fig)

# Footer
st.markdown("""
    <hr>
    <div class="footer">
        🚀 Developed by <b>Rohith Obillaneni</b> | Powered by Machine Learning & FastAPI
    </div>
""", unsafe_allow_html=True)
