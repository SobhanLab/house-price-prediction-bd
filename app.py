# 1. Import
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


# 2. App Config
st.set_page_config(page_title="House Price Predictor", page_icon="🏠")


# 3. Title
st.title("🏠 House Price Prediction (BD)")
st.info("Model Performance: R² ≈ 0.836")
st.write("Enter house details to estimate price")


# 4. Load & Train (cached)
@st.cache_resource
def load_model():
    df = pd.read_csv("house_price_bd.csv")

    df['Price_in_taka'] = df['Price_in_taka'].str.replace('৳', '')
    df['Price_in_taka'] = df['Price_in_taka'].str.replace(',', '')
    df['Price_in_taka'] = df['Price_in_taka'].astype(float)

    df = df.dropna()
    df = df.drop_duplicates()
    df = df.drop('Title', axis=1)

    X = df[['Bedrooms','Bathrooms','Floor_area','City','Location']]
    y = np.log1p(df['Price_in_taka'])

    preprocessor = ColumnTransformer([
        ('num','passthrough',['Bedrooms','Bathrooms','Floor_area']),
        ('cat',OneHotEncoder(handle_unknown='ignore'),['City','Location'])
    ])

    pipeline = Pipeline([
        ('prep',preprocessor),
        ('model',XGBRegressor(n_estimators=200))
    ])

    pipeline.fit(X, y)
    return pipeline


pipeline = load_model()


# 5. Inputs
bedrooms = st.slider("Bedrooms", 1, 10, 3)
bathrooms = st.slider("Bathrooms", 1, 10, 2)
area = st.number_input("Floor Area (sqft)", 500, 10000, 1500)
city = st.selectbox("City", ["dhaka"])
location = st.text_input("Location (e.g., Gulshan, Dhanmondi, Mirpur)")


# 6. Validation
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
