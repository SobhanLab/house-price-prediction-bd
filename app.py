import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

st.title("🏠 House Price Prediction (BD)")

@st.cache_data
def load_data():
    df = pd.read_csv("house_price_bd.csv")

    df['Price_in_taka'] = df['Price_in_taka'].str.replace('৳', '')
    df['Price_in_taka'] = df['Price_in_taka'].str.replace(',', '')
    df['Price_in_taka'] = df['Price_in_taka'].astype(float)

    df = df.dropna()
    df = df.drop_duplicates()
    df = df.drop('Title', axis=1)

    return df

@st.cache_resource
def train_model(df):
    X = df[['Bedrooms', 'Bathrooms', 'Floor_area']]
    y = np.log1p(df['Price_in_taka'])

    model = XGBRegressor(
        n_estimators=80,
        max_depth=4,
        learning_rate=0.1
    )
    model.fit(X, y)

    return model

# Load once
df = load_data()
model = train_model(df)

# UI
bedrooms = st.slider("Bedrooms", 1, 10, 3)
bathrooms = st.slider("Bathrooms", 1, 10, 2)
area = st.number_input("Area (sqft)", 500, 10000, 1500)

if st.button("Predict Price"):
    input_data = np.array([[bedrooms, bathrooms, area]])
    pred = model.predict(input_data)
    price = np.expm1(pred[0])

    st.success(f"Estimated Price: {price:,.0f} BDT")
