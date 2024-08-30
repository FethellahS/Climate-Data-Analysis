import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from data_preprocessing import load_data, preprocess_data

def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

def lstm_forecast(data):
    values = data.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(values)

    # Convert the data to sequences
    time_step = 12  # Number of months to look back
    X, y = create_dataset(scaled_values, time_step)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape for LSTM

    # Define the model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X, y, epochs=50, batch_size=1, verbose=2)

    # Forecast
    test_X = scaled_values[-time_step:].reshape((1, time_step, 1))
    forecast = model.predict(test_X)
    forecast = scaler.inverse_transform(forecast)  # Inverse transform to original scale

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(data, label='Historical Data')
    plt.plot(pd.date_range(start=data.index[-1], periods=2, freq='M')[1:], forecast[0], label='Forecast', color='red')
    plt.legend()
    plt.title('LSTM Forecast')
    plt.show()

if __name__ == "__main__":
    data = load_data('climate_data.csv')
    data = preprocess_data(data)
    lstm_forecast(data)
