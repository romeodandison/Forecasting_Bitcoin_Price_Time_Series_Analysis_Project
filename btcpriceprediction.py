
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the pre-trained model
model = load_model('btcpriceforecast.h5')  # Make sure to save your model as 'your_model.h5'

# Function to fetch and preprocess data
def fetch_data():
    ticker_symbol = "BTC-USD"
    crypto_data = yf.download(ticker_symbol, start="2009-01-03", end="2024-11-17", interval="1d")
    return crypto_data

# Function to make predictions
def predict_future_prices(data, model, days=7):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    last_sequence = scaled_data[-60:].reshape(1, -1, 1)  # Reshape for LSTM
    future_prices = []
    
    for _ in range(days):
        next_price = model.predict(last_sequence)
        future_prices.append(next_price[0, 0])
        last_sequence = np.append(last_sequence[:, 1:, :], next_price.reshape(1, 1, 1), axis=1)

    future_prices = scaler.inverse_transform(np.array(future_prices).reshape(-1, 1))
    return future_prices.flatten()

# Streamlit UI
st.title("Bitcoin Price Prediction")
st.write("Predict future Bitcoin prices using LSTM model.")

# Fetch data
data = fetch_data()

# Show the latest data
st.subheader("Latest Bitcoin Data")
st.write(data.tail())

# Prediction
if st.button('Predict Next 7 Days'):
    future_prices = predict_future_prices(data, model)
    st.subheader("Predicted Prices for Next 7 Days")
    st.write(future_prices)

    # Plotting the future prices
    plt.figure(figsize=(10, 5))
    plt.plot(future_prices, marker='o', color='red', label='Predicted Price')
    plt.title('Bitcoin Price Prediction for the Next 7 Days')
    plt.xlabel('Days')
    plt.ylabel('Price in USD')
    plt.legend()
    st.pyplot(plt)   
