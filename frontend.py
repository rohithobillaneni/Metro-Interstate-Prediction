import streamlit as st
import requests
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
st.markdown("<p class='sub-title'>Utilising machine learning to predict traffic congestion based on weather, time, and other factors.</p>", unsafe_allow_html=True)
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
hour_category = st.sidebar.selectbox("🕒 Time of Day", ['Early Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late Night'])
month = st.sidebar.selectbox("📅 Month", ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])

# Predict Button (Centered)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🚀 Predict Traffic Volume")

# Prediction Logic
if predict_button:
    with st.spinner("🔄 Predicting..."):
        # API request
        api_url = "http://traffic-backend:8000/predict"
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
        response = requests.post(api_url, json=data)

        # Handle response
        if response.status_code == 200:
            prediction = response.json().get("prediction", "Error")
            
            # Display Results
            st.markdown("### 🚗 **Predicted Traffic Volume**")
            if isinstance(prediction, (int, float)):  
                st.success(f"🔹 **{int(prediction)} vehicles**")

                # Visualization - Traffic Volume Gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=int(prediction),
                    title={'text': "Traffic Volume"},
                    gauge={'axis': {'range': [None, 10000]},
                           'bar': {'color': "#3498db"},
                           'steps': [
                               {'range': [0, 3000], 'color': "#2ecc71"},
                               {'range': [3000, 7000], 'color': "#f1c40f"},
                               {'range': [7000, 10000], 'color': "#e74c3c"}
                           ]}
                ))
                st.plotly_chart(fig)

            else:
                st.error("⚠️ Invalid prediction received. Please check the API response.")
        else:
            st.error("❌ API request failed. Ensure FastAPI is running.")

# Footer
st.markdown("""
    <hr>
    <div class="footer">
        🚀 Developed by <b>Rohith Obillaneni</b> | Powered by Machine Learning & FastAPI
    </div>
""", unsafe_allow_html=True)
