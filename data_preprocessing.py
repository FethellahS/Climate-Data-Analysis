import pandas as pd

def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    return data

def preprocess_data(data):
    # Example of resampling to monthly data if needed
    data = data.resample('M').mean()
    data = data.fillna(method='ffill')  # Forward fill to handle missing data
    return data
