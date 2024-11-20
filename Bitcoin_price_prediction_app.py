
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

st.title("Bitcoin Price Prediction using LGBM")

# Load dataset
crypto_data = pd.read_csv('Documents/Data Science/Project/crypto_data_final.csv')

# Feature engineering
crypto_data['Date'] = pd.to_datetime(crypto_data['Date'])
crypto_data.set_index('Date', inplace=True)
crypto_data['day'] = crypto_data.index.day
crypto_data['month'] = crypto_data.index.month
crypto_data['year'] = crypto_data.index.year

# Scale features
scaler = StandardScaler()
crypto_data[['Close', 'High', 'Low', 'Open', 'Volume']] = scaler.fit_transform(crypto_data[['Close', 'High', 'Low', 'Open', 'Volume']])

# Prepare training data
X = crypto_data.drop('Close', axis=1)
y = crypto_data['Close']

# Load the trained LGBM model
lgbm_model = lgb.Booster(model_file='Documents/Data Science/Project/lgbm_bitcoin_price_model.txt')

st.write("Predict the Bitcoin Price")
date_input = st.date_input("Select a date", min_value=crypto_data.index.min(), max_value=crypto_data.index.max())
selected_date = pd.to_datetime(date_input)

# Get input data for prediction
input_data = crypto_data.loc[selected_date]

# Align features to match training
input_data = input_data[X.columns]

# Predict
prediction = lgbm_model.predict(input_data.values.reshape(1, -1))
st.write(f"Predicted Bitcoin Price for {selected_date.strftime('%Y-%m-%d')}: ${prediction[0]:.2f}")
