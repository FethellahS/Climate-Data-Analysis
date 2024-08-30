import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from data_preprocessing import load_data, preprocess_data

def arima_forecast(data):
    # Fit the ARIMA model
    model = ARIMA(data, order=(5,1,0))  # Adjust the order parameters as needed
    model_fit = model.fit(disp=0)

    # Forecast
    forecast = model_fit.forecast(steps=12)  # Forecast for the next 12 months

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(data, label='Historical Data')
    plt.plot(pd.date_range(start=data.index[-1], periods=13, freq='M')[1:], forecast[0], label='Forecast', color='red')
    plt.legend()
    plt.title('ARIMA Forecast')
    plt.show()

if __name__ == "__main__":
    data = load_data('climate_data.csv')
    data = preprocess_data(data)
    arima_forecast(data)
