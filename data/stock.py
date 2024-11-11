import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Tuple, Union
from sklearn.preprocessing import MinMaxScaler

def download_stock_data(symbol: str, start: str = '2000-01-01', end: str = None) -> pd.DataFrame:
    """
    Downloads daily S&P 500 data within a specified date range, cleans it, and computes log returns.

    Parameters
    ------------
    start : str, optional
        Start date in 'YYYY-MM-DD' format. Default is '2000-01-01'.
    end : str, optional
        End date in 'YYYY-MM-DD' format. Default is the current date.

    Returns
    ---------
    sp500 : pd.DataFrame
        Cleaned DataFrame containing S&P 500 data with log returns.
    """
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')

    # Download S&P 500 data from Yahoo Finance
    stock = yf.download(symbol, start=start, end=end, progress=False)

    # Reset index to get 'Date' as a column
    stock.reset_index(inplace=True)

    # Flatten MultiIndex columns if they exist
    if isinstance(stock.columns, pd.MultiIndex):
        stock.columns = stock.columns.get_level_values(0)

    # Clean 'Date' column
    stock['Date'] = pd.to_datetime(stock['Date'], utc=True).dt.tz_convert(None).dt.normalize()
    stock = stock.set_index('Date')

    # # Calculate log returns from 'Adj Close' price
    # stock['Log_Returns'] = np.log(stock['Adj Close'] / stock['Adj Close'].shift(1))

    # # Drop NaN values resulting from the shift
    # stock.dropna(subset=['Log_Returns'], inplace=True)

    return stock

def phase_space_reconstruction(series, delay, embedding_dim):
    N, num_features = series.shape
    reconstructed = np.zeros((N - (embedding_dim - 1) * delay, embedding_dim * num_features))
    for i in range(num_features):
        for d in range(embedding_dim):
            reconstructed[:, i * embedding_dim + d] = series[d * delay: N - (embedding_dim - d - 1) * delay, i]
    return reconstructed

def preprocess_data(data, delay, embedding_dim, scaler_features=None, scaler_target=None):
    # Separate features and target
    features = data.loc[:, data.columns != 'Close']  # All columns except 'Close'
    target = data.loc[:, 'Close'].values.reshape(-1, 1)  # Just the 'Close' column

    # Initialize and fit/transform scalers
    if scaler_features is None:
        scaler_features = MinMaxScaler()
        features = scaler_features.fit_transform(features)
    else:
        features = scaler_features.transform(features)

    if scaler_target is None:
        scaler_target = MinMaxScaler()
        target = scaler_target.fit_transform(target)
    else:
        target = scaler_target.transform(target)

    # Apply phase space reconstruction
    reconstructed_features = phase_space_reconstruction(features, delay, embedding_dim)
    target = target[delay * (embedding_dim - 1):]

    return reconstructed_features, target.squeeze(), scaler_features, scaler_target

def reconstruct_sequences(sequences, step=1):
    """
    Reconstructs a time series from overlapping sequences.

    Parameters:
    -----------
    sequences : np.ndarray
        Array of shape (num_sequences, sequence_length, 1)
    step : int
        Step size used in the sliding window.

    Returns:
    --------
    time_series : np.ndarray
        Reconstructed time series of shape (total_length, 1)
    """
    num_sequences, sequence_length, _ = sequences.shape
    total_length = (num_sequences - 1) * step + sequence_length
    time_series = np.zeros((total_length, 1))
    counts = np.zeros((total_length, 1))

    for i in range(num_sequences):
        start = i * step
        end = start + sequence_length
        time_series[start:end] += sequences[i, :, 0].reshape(-1, 1)
        counts[start:end] += 1

    # Avoid division by zero
    counts[counts == 0] = 1

    time_series = time_series / counts
    return time_series