# 1. Import
import streamlit as st
import pandas as pd
import numpy as np
import pickle


# 2. App Config
st.set_page_config(page_title="House Price Predictor", page_icon="🏠")


# 3. Title & Info
st.title("🏠 House Price Prediction (BD)")
st.info("Model Performance: R² ≈ 0.836")
st.write("Enter house details to estimate price")


# 4. Load Pipeline
@st.cache_resource
def load_model():
    return pickle.load(open("app.pkl", "rb"))

pipeline = load_model()


# 5. User Inputs
bedrooms = st.slider("Bedrooms", 1, 10, 3)
bathrooms = st.slider("Bathrooms", 1, 10, 2)
area = st.number_input("Floor Area (sqft)", 500, 10000, 1500)

city = st.selectbox("City", ["dhaka"])
location = st.text_input("Location (e.g., Gulshan, Dhanmondi, Mirpur)")


# 6. Input Validation
if bedrooms * 200 > area:
    st.warning("Area too small for given number of bedrooms")


# 7. Prediction
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

    st.success(f"Estimated Price: {price:,.0f} BDT")
