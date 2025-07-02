import streamlit as st
import numpy as np
import pickle

# Load model
try:
    with open('crop_model.pkl', 'rb') as f:
        model = pickle.load(f)
    model_status = True
except:
    model_status = False

st.title("🌾 AI Assistant for Smart Farming")

st.subheader("📋 Enter Crop & Soil Information")

# Inputs
N = st.number_input("Nitrogen (N)", min_value=0)
P = st.number_input("Phosphorus (P)", min_value=0)
K = st.number_input("Potassium (K)", min_value=0)
temperature = st.slider("Temperature (°C)", 10.0, 50.0)
humidity = st.slider("Humidity (%)", 10.0, 100.0)
ph = st.slider("pH Level", 0.0, 14.0)
rainfall = st.slider("Rainfall (mm)", 0.0, 300.0)

if st.button("🌱 Predict Best Crop"):
    if model_status:
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = model.predict(input_data)[0]
        st.success(f"✅ Recommended Crop: **{prediction}**")
    else:
        st.error("❌ Model not found or failed to load.")

st.markdown("---")
st.markdown("📊 [View Soil & Weather Info Tips](#)")
st.markdown("Made with ❤️ by AI Helper For Farmer Team | NSTI Noida")
