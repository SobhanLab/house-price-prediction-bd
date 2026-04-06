import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

st.title("🏠 House Price Prediction (BD)")
st.write("Enter house details to estimate price")

# Inputs
bedrooms = st.slider("Bedrooms", 1, 10, 3)
bathrooms = st.slider("Bathrooms", 1, 10, 2)
area = st.number_input("Floor Area (sqft)", 500, 10000, 1500)

# Predict
if st.button("Predict Price"):
    input_data = np.array([[bedrooms, bathrooms, area]])
    
    prediction = model.predict(input_data)
    
    st.success(f"Estimated Price: {prediction[0]:,.0f} BDT")
