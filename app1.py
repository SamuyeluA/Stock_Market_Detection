import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from datetime import datetime
from keras.models import load_model  # Assuming you have a pre-trained model

st.title('Stock Market Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')

# Get user input for start and end dates
start_date_str = st.text_input('Start Date (YYYY-MM-DD)', '2010-01-01')
end_date_str = st.text_input('End Date (YYYY-MM-DD)', '2022-12-31')

try:
    # Convert date strings to datetime objects
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

    # Fetch data using yfinance
    df = yf.download(user_input, start=start_date, end=end_date)

    # Describe the data
    st.subheader('Data Description')
    st.write(df.describe())

except Exception as e:
    st.error(f"Error retrieving data: {e}")

def plot_data(data, title):
    fig = plt.figure(figsize=(12, 6))
    plt.plot(data)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    st.pyplot(fig)

st.subheader('Closing Price vs Time')
plot_data(df.Close, 'Closing Price')

def plot_ma(data, window, title):
    ma = data.Close.rolling(window).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df.Close, label='Original Price')
    plt.plot(ma, label=f'{window}-Day Moving Average')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig)

plot_ma(df.copy(), 100, 'Closing Price vs 100-Day Moving Average')
plot_ma(df.copy(), 200, 'Closing Price vs 200-Day Moving Average')

# Splitting data into training and testing
data_training = pd.DataFrame(df["Close"][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df["Close"][int(len(df)*0.70): int(len(df))])

scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Load the pre-trained model
try:
    model = load_model('keras_model.h5')
except FileNotFoundError:
    st.error("Error: Pre-trained model 'keras_model.h5' not found.")
    st.stop()

# Testing Part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

def plot_prediction(original, predicted, title):
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(original, "b", label="Original Price")
    plt.plot(predicted, "r", label="Predicted Price")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(fig2)

st.subheader('Prediction vs Original')
plot_prediction(y_test, y_predicted, 'Predicted vs Actual Price')