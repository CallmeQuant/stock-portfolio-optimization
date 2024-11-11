import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Tuple, Union

def download_sp500_data(start: str = '2000-01-01', end: str = None) -> pd.DataFrame:
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
    sp500 = yf.download('^GSPC', start=start, end=end, progress=False)

    # Reset index to get 'Date' as a column
    sp500.reset_index(inplace=True)

    # Flatten MultiIndex columns if they exist
    if isinstance(sp500.columns, pd.MultiIndex):
        sp500.columns = sp500.columns.get_level_values(0)

    # Clean 'Date' column
    sp500['Date'] = pd.to_datetime(sp500['Date'], utc=True).dt.tz_convert(None).dt.normalize()

    # Calculate log returns from 'Adj Close' price
    sp500['Log_Returns'] = np.log(sp500['Adj Close'] / sp500['Adj Close'].shift(1))

    # Drop NaN values resulting from the shift
    sp500.dropna(subset=['Log_Returns'], inplace=True)

    return sp500

def create_sequences(
    returns: Union[pd.Series, np.ndarray],
    sequence_length: int = 20,
    step: int = 1,
    train_test_split: float = 0.8,
    shuffle: bool = True,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates sequences of specified length from log returns data, with optional train-test split.

    Parameters
    ----------
    returns : Union[pd.Series, np.ndarray]
        Log returns series, shape (n_samples, 1) or (n_samples,)
    sequence_length : int, optional
        Length of each sequence, default is 20
    step : int, optional
        Step size between sequences, default is 1
    train_test_split : float, optional
        Proportion of data to use for training, default is 0.8
    shuffle : bool, optional
        Whether to shuffle the sequences, default is True
    seed : int, optional
        Random seed for reproducibility, default is 42

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Training and testing sequences with shapes (n_train, sequence_length, 1)
        and (n_test, sequence_length, 1) respectively
    """
    # Convert to numpy array if pandas Series
    if isinstance(returns, pd.Series):
        returns = returns.values

    # Ensure returns is 1D array
    returns = returns.reshape(-1)

    # Create sequences
    sequences = []
    for i in range(0, len(returns) - sequence_length + 1, step):
        sequence = returns[i:(i + sequence_length)]
        sequences.append(sequence)

    # Convert to numpy array and reshape
    sequences = np.array(sequences)
    sequences = sequences.reshape(-1, sequence_length, 1)

    # Shuffle if requested
    if shuffle:
        np.random.seed(seed)
        shuffle_idx = np.random.permutation(len(sequences))
        sequences = sequences[shuffle_idx]

    # Split into train and test sets
    split_idx = int(len(sequences) * train_test_split)
    train_sequences = sequences[:split_idx]
    test_sequences = sequences[split_idx:]

    return train_sequences, test_sequences

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

def prepare_sp500_sequences(
    start: str = '2000-01-01',
    end: str = None,
    sequence_length: int = 20,
    step: int = 1,
    train_test_split: float = 0.8,
    shuffle: bool = False,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downloads S&P 500 data and prepares sequences for training.

    Parameters
    ----------
    start : str, optional
        Start date in 'YYYY-MM-DD' format
    end : str, optional
        End date in 'YYYY-MM-DD' format
    sequence_length : int, optional
        Length of each sequence
    step : int, optional
        Step size between sequences
    train_test_split : float, optional
        Proportion of data to use for training
    shuffle : bool, optional
        Whether to shuffle the sequences
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Training and testing sequences
    """
    # Download and prepare S&P 500 data
    sp500_df = download_sp500_data(start=start, end=end)

    # Create sequences from log returns
    train_sequences, test_sequences = create_sequences(
        returns=sp500_df['Log_Returns'],
        sequence_length=sequence_length,
        step=step,
        train_test_split=train_test_split,
        shuffle=shuffle,
        seed=seed
    )

    return train_sequences, test_sequences, sp500_df