import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

st.title("🏠 House Price Prediction (BD)")
st.info("Model Performance: R² ≈ 0.836")
st.write("Enter house details to estimate price")

@st.cache_resource
def load_model():
    with open("pipeline.pkl", "rb") as f:
        return pickle.load(f)

pipeline = load_model()

bedrooms = st.slider("Bedrooms", 1, 10, 3)
bathrooms = st.slider("Bathrooms", 1, 10, 2)
area = st.number_input("Floor Area (sqft)", 500, 10000, 1500)

city = st.selectbox("City", ["dhaka"])
location = st.text_input("Location (e.g., Gulshan, Dhanmondi, Mirpur)")

if bedrooms * 200 > area:
    st.warning("Area too small for given number of bedrooms")

if st.button("Predict Price"):
    input_df = pd.DataFrame({
        "Bedrooms": [bedrooms],
        "Bathrooms": [bathrooms],
        "Floor_area": [area],
        "City": [city],
        "Location": [location]
    })

    pred = pipeline.predict(input_df)
    price = np.expm1(pred[0])

    st.success(f"💰 Estimated Price: {price:,.0f} BDT")
