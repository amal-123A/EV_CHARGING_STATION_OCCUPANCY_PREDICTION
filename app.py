import streamlit as st
import numpy as np
import joblib

rf = joblib.load("ev_rf_model.pkl")
scaler = joblib.load("ev_scaler.pkl")

st.set_page_config(page_title="EV Occupancy Prediction", layout="centered")
st.title("🔌 EV Charging Station Occupancy Prediction")

st.markdown("Enter station details to predict occupancy status")


station_id = st.number_input("Station ID", min_value=1, max_value=10000, value=100)

city_name = st.selectbox(
    "City",
    ["Chennai", "Bangalore", "Hyderabad", "Mumbai"]
)
city_map = {
    "Chennai": 0,
    "Bangalore": 1,
    "Hyderabad": 2,
    "Mumbai": 3
}
city = city_map[city_name]

day_name = st.selectbox("Day Type", ["Weekday", "Weekend"])
day_map = {"Weekday": 0, "Weekend": 1}
day_type = day_map[day_name]

hour = st.slider("Hour of Day", 0, 23, 18)

chargers = st.number_input("Number of Chargers", min_value=1, max_value=50, value=6)

past_rate = st.slider("Past Occupancy Rate", 0.0, 1.0, 0.85)

weather_name = st.selectbox(
    "Weather",
    ["Clear", "Rainy", "Cloudy"]
)
weather_map = {"Clear": 0, "Rainy": 1, "Cloudy": 2}
weather = weather_map[weather_name]


if st.button("🔍 Predict Occupancy"):

    input_data = np.array([[ 
        station_id,
        city,
        day_type,
        hour,
        chargers,
        past_rate,
        weather
    ]])

    st.write("Input shape:", input_data.shape)
    st.write("Scaler expects:", scaler.n_features_in_)

    input_scaled = scaler.transform(input_data)

    prediction = rf.predict(input_scaled)
    probability = rf.predict_proba(input_scaled)[0][1]

    st.subheader("📊 Prediction Result")

    if probability >= 0.5:
        st.error(f"🚗 Station is BUSY (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Station is FREE (Probability: {1-probability:.2f})")