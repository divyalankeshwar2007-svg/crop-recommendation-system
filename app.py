import streamlit as st
import pandas as pd
import joblib
import time

# Page config
st.set_page_config(page_title="Crop Recommendation System", page_icon="🌾", layout="centered")

# Banner image
st.image(
    "https://images.unsplash.com/photo-1500382017468-9049fed747ef",
    width="stretch"
)

# Title
st.title("🌾 Crop Recommendation System")
st.markdown("**Helping the Farmers in Maharashtra to Choose the Best Crop**")

# Load model and encoder
@st.cache_resource
def load_all():
    model = joblib.load("model.pkl")   # ✅ pipeline model
    le_target = joblib.load("le_target.pkl")
    return model, le_target

model, le_target = load_all()

# Features
numeric_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# Input section
st.subheader("Enter the Soil and Weather Conditions")

col1, col2, col3 = st.columns(3)

with col1:
    N = st.number_input("Nitrogen (N)", 0, 140, 50)
    P = st.number_input("Phosphorus (P)", 0, 145, 50)
    K = st.number_input("Potassium (K)", 0, 205, 50)

with col2:
    temperature = st.number_input("Temperature (°C)", 8.0, 43.0, 25.0)
    humidity = st.number_input("Humidity (%)", 14.0, 100.0, 70.0)
    ph = st.number_input("pH Value", 3.5, 9.9, 6.5)

with col3:
    rainfall = st.number_input("Rainfall (mm)", 20.0, 300.0, 100.0)

# Crop images
crop_images = {
    "rice": "https://images.unsplash.com/photo-1586201375761-83865001e31c",
    "wheat": "https://images.unsplash.com/photo-1500382017468-9049fed747ef",
    "maize": "https://images.unsplash.com/photo-1601597111158-2fceff292cdc",
    "cotton": "https://images.unsplash.com/photo-1592928302636-c83cf1c7b2c7",
    "sugarcane": "https://images.unsplash.com/photo-1625246333195-78d9c38ad449",
    "coffee": "https://images.unsplash.com/photo-1509042239860-f550ce710b93",
    "jute": "https://images.unsplash.com/photo-1598514982841-2c2c9f9f9f9f",
    "papaya": "https://images.pexels.com/photos/4113807/pexels-photo-4113807.jpeg",
    "banana": "https://upload.wikimedia.org/wikipedia/commons/8/8a/Banana-Single.jpg",
    "mango": "https://images.unsplash.com/photo-1553279768-865429fa0078",
    "apple": "https://images.unsplash.com/photo-1567306226416-28f0efdc88ce",
    "grapes": "https://images.unsplash.com/photo-1596363505729-4190a9506133",
    "orange": "https://images.unsplash.com/photo-1580052614034-c55d20bfee3b",
    "pigeonpeas": "https://upload.wikimedia.org/wikipedia/commons/0/0c/Pigeon_pea.jpg",
}

# Predict button
if st.button("🌱 Predict Best Crop", type="primary", width="stretch"):

    with st.spinner("🌱 Analyzing soil and weather data..."):
        time.sleep(1)

        input_data = pd.DataFrame(
            [[N, P, K, temperature, humidity, ph, rainfall]],
            columns=numeric_features
        )
        
        prediction = model.predict(input_data)
        crop_name = prediction[0]

    st.success(f"**Recommended Crop: {crop_name.upper()}** 🌾")

    st.markdown("### 🌿 Recommended Crop Preview")

    image_url = crop_images.get(crop_name.lower())

    if image_url:
        st.image(image_url, caption=f"🌾 {crop_name.upper()}", width="stretch")
    else:
        st.image("https://picsum.photos/id/1015/400/300", caption=crop_name.upper(), width="stretch")
