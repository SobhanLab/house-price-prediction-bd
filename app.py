import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

st.title("🏠 House Price Prediction (BD)")

df = pd.read_csv("house_price_bd.csv")

df['Price_in_taka'] = df['Price_in_taka'].str.replace('৳', '')
df['Price_in_taka'] = df['Price_in_taka'].str.replace(',', '')
df['Price_in_taka'] = df['Price_in_taka'].astype(float)

df = df.dropna()
df = df.drop_duplicates()
df = df.drop('Title', axis=1)

X = df.drop('Price_in_taka', axis=1)
y = np.log1p(df['Price_in_taka'])

X = pd.get_dummies(X, drop_first=True)

model = XGBRegressor(n_estimators=200)
model.fit(X, y)

bedrooms = st.slider("Bedrooms", 1, 10, 3)
bathrooms = st.slider("Bathrooms", 1, 10, 2)
area = st.number_input("Area", 500, 10000, 1500)

if st.button("Predict"):
    input_data = np.array([[bedrooms, bathrooms, area]])
    pred = model.predict(input_data)
    price = np.expm1(pred[0])
    
    st.success(f"Estimated Price: {price:,.0f} BDT")
