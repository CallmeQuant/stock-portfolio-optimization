import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim.lr_scheduler import OneCycleLR
from torchdiffeq import odeint_adjoint as odeint
import copy
import matplotlib.pyplot as plt
import streamlit as st 
from data.stock import download_stock_data

from mapie.metrics import regression_coverage_score, regression_mean_width_score, coverage_width_based
from mapie.subsample import BlockBootstrap
from mapie.regression import MapieTimeSeriesRegressor
from mapie.conformity_scores.regression import BaseRegressionScore

def phase_space_reconstruction(series, delay, embedding_dim):
    N, num_features = series.shape
    reconstructed = np.zeros((N - (embedding_dim - 1) * delay, embedding_dim * num_features))
    for i in range(num_features):
        for d in range(embedding_dim):
            reconstructed[:, i * embedding_dim + d] = series[d * delay: N - (embedding_dim - d - 1) * delay, i]
    return reconstructed

def preprocess_data(data, delay, embedding_dim, scaler_features=None, scaler_target=None):
    # Separate features and target
    features = data.loc[:, ~data.columns.isin(['Close'])]  # All columns except 'Close'
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

class ODEFunc(nn.Module):
    def __init__(self, dim):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 50),  # 'dim' should match the number of features in data
            nn.Tanh(),
            nn.Linear(50, dim),
        )

    def forward(self, t, y):
        return self.net(y)

class NeuralODEModel(nn.Module):
    def __init__(self, ode_func, num_features, h_steps=1):
        super(NeuralODEModel, self).__init__()
        self.ode_func = ode_func
        self.linear = nn.Linear(num_features, h_steps) 

    def forward(self, x):
        ode_result = odeint(self.ode_func, x, torch.tensor([0., 1.]))[1]
        return self.linear(ode_result)

def train_neural_ode_with_validation(model, train_data, train_target,
                                     val_data, val_target, epochs, lr, patience):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_tensor = torch.from_numpy(train_data).float()
    train_target_tensor = torch.from_numpy(train_target).float().squeeze()
    val_tensor = torch.from_numpy(val_data).float()
    val_target_tensor = torch.from_numpy(val_target).float().squeeze()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        prediction = model(train_tensor).squeeze()
        loss = criterion(prediction, train_target_tensor)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_prediction = model(val_tensor).squeeze()
            val_loss = criterion(val_prediction, val_target_tensor)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            print(f'Early stopping triggered at epoch {epoch}')
            break

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Train Loss: {loss.item()}, Val Loss: {val_loss.item()}')

    model.load_state_dict(best_model_wts)
    return model

def forecast_stock_prices(symbol, start_date, end_date, context_window=50, 
                          h_steps = 5,
                          alpha = 0.05,
                          partial_fit = False):
    """Main function to forecast stock prices"""
    try:
        # Download and prepare data
        raw_data = download_stock_data(symbol, start_date, end_date)
        dates = pd.date_range(start=start_date, end=end_date)

        # Split data
        split_idx = int(len(raw_data) * 0.8)
        train_data, test_data = raw_data[:split_idx], raw_data[split_idx:]
        test_dates = test_data.index

        # Preprocess data
        train_features, train_target, scaler_features, scaler_target = preprocess_data(train_data, 2, 4)
        test_features, test_target, _, _ = preprocess_data(test_data, 2, 4, scaler_features, scaler_target)
        
        # Initialize model
        num_features_reconstructed = train_features.shape[1]
        neural_ode_model = NeuralODEModel(ODEFunc(dim=num_features_reconstructed), num_features_reconstructed)
        
        # Train model with cross-validation
        tscv = TimeSeriesSplit(n_splits=4)
        for train_index, val_index in tscv.split(train_features):
            kf_train_features = train_features[train_index]
            kf_val_features = train_features[val_index]
            kf_train_target = train_target[train_index]
            kf_val_target = train_target[val_index]
            
            neural_ode_model = train_neural_ode_with_validation(
                neural_ode_model, 
                kf_train_features, 
                kf_train_target,
                kf_val_features,
                kf_val_target,
                epochs=100,
                lr=0.001,
                patience=10
            )
        
        # Initiate Mapie 
        print("Initializing MAPIE Wrapper...")
        wrapped_model = MappieWrapper(neural_ode_model)

        print("Setting up cross-validation strategy...")
        h = 1
        cv_mapiets = BlockBootstrap(
            n_resamplings=30, length = 10, overlapping=True, random_state=88
        )
        mapie_aci = MapieTimeSeriesRegressor(
            wrapped_model, method="aci", cv=cv_mapiets, agg_function="mean", n_jobs=-1
        )
        print("Fitting MAPIE on training data...")
        mapie_aci.fit(train_features, train_target)
        print("Model fitting complete.")

        print("Generating prediction intervals on test data...")
        y_pred, y_pis = mapie_aci.predict(test_features, alpha=alpha, 
                                        ensemble=True, 
                                        allow_infinite_bounds=True)
        if partial_fit:
            # Allow new residuals to adapt forecast intervals
            y_pred_pfit, y_pis_pfit = np.zeros(y_pred.shape), np.zeros(y_pis.shape)
            y_pred_pfit[:h], y_pis_pfit[:h, :, :] = mapie_aci.predict(test_features[:h, :],
                                                                 alpha=alpha,
                                                                 ensemble=True,
                                                                 allow_infinite_bounds=True)     
            for step in range(h, len(test_features), h):
                mapie_aci.partial_fit(test_features[(step-h): step, :],
                                        test_target[(step-h):step])
    
                y_pred_pfit[step:step + h], \
                    y_pis_pfit[step:step + h, :, :] = mapie_aci.predict(test_features[step:(step+h), :],
                                                                                           alpha=alpha,
                                                                                           ensemble=True,
                                                                                           allow_infinite_bounds=True)  
        if partial_fit:         
            coverage = regression_coverage_score(
                    test_target, y_pis_pfit[:, 0, 0], y_pis_pfit[:, 1, 0]
                )
            width = regression_mean_width_score(
                    y_pis_pfit[:, 0, 0], y_pis_pfit[:, 1, 0]
                )
            cwc = coverage_width_based(
                    test_target, y_pis_pfit[:, 0, 0], y_pis_pfit[:, 1, 0], eta = 10, alpha = alpha
                )
        else:
            coverage = regression_coverage_score(
                    test_target, y_pis[:, 0, 0], y_pis[:, 1, 0]
                )
            width = regression_mean_width_score(
                    y_pis[:, 0, 0], y_pis[:, 1, 0]
                )
            cwc = coverage_width_based(
                    test_target, y_pis[:, 0, 0], y_pis[:, 1, 0], eta = 10, alpha = alpha
                )

        if partial_fit:                                                                                
            # Inverse transform predictions
            predicted_prices = scaler_target.inverse_transform(y_pred_pfit.reshape(-1, 1))
            lower_bound = scaler_target.inverse_transform(y_pis_pfit[:, 0, 0].reshape(-1, 1))
            upper_bound = scaler_target.inverse_transform(y_pis_pfit[:, 1, 0].reshape(-1, 1))
            actual_prices = scaler_target.inverse_transform(test_target.reshape(-1, 1))
            print("In-sample forecast completed successfully!. Start out-of-sample forecast.")
        else:
            # Inverse transform predictions
            predicted_prices = scaler_target.inverse_transform(y_pred.reshape(-1, 1))
            lower_bound = scaler_target.inverse_transform(y_pis[:, 0, 0].reshape(-1, 1))
            upper_bound = scaler_target.inverse_transform(y_pis[:, 1, 0].reshape(-1, 1))
            actual_prices = scaler_target.inverse_transform(test_target.reshape(-1, 1))
            print("In-sample forecast completed successfully!. Start out-of-sample forecast.")

        # After getting in-sample predictions, add the future forecasts
        future_pred, future_lower, future_upper, future_dates = forecast_future_steps(
            test_features, 
            test_target,   
            neural_ode_model,
            mapie_aci,
            test_data[-context_window:], 
            scaler_features,
            scaler_target,
            h_steps=h_steps,
            delay=2,
            embedding_dim=4,
            alpha=alpha
        )
        print("Out-of-sample forecast completed sucessfully! ")
        return (predicted_prices, actual_prices, lower_bound, upper_bound, 
                test_dates, coverage, width, cwc,
                future_pred, future_lower, future_upper, future_dates,
                train_data, test_data)
    except Exception as e:
        st.error(f"Error in forecasting: {str(e)}")
        return None, None, None, None, None, None, None, None, None, None, None, None

def forecast_future_steps(test_features, test_target, model, mapie_model, 
                         last_known_data, scaler_features, scaler_target, 
                         h_steps, delay, embedding_dim, alpha=0.05):
    """
    Perform h-step ahead forecasting with prediction intervals starting from last in-sample forecast
    
    Args:
        test_features: Features used for the last in-sample prediction
        test_target: Target values for the test set
        model: Trained Neural ODE model
        mapie_model: Trained MAPIE model
        last_known_data: DataFrame with the last known data points (context window)
        scaler_features: Fitted feature scaler
        scaler_target: Fitted target scaler
        h_steps: Number of steps to forecast ahead
        delay: Delay parameter used in phase space reconstruction
        embedding_dim: Embedding dimension used in phase space reconstruction
        alpha: Significance level for prediction intervals
        
    Returns:
        tuple: (predictions, lower_bounds, upper_bounds, forecast_dates)
    """
    # Initialize storage for predictions and intervals
    future_predictions = []
    future_lower_bounds = []
    future_upper_bounds = []
    
    # Get the last in-sample prediction
    with torch.no_grad():
        last_pred, last_pis = mapie_model.predict(
            test_features[-1:],
            alpha=alpha,
            ensemble=True
        )
    
    # Create a copy of the context window data
    current_data = last_known_data.copy()
    
    # Add the last prediction to the context
    new_row = current_data.iloc[-1:].copy()
    new_row.index = new_row.index + pd.Timedelta(days=1)
    new_row['Close'] = scaler_target.inverse_transform(last_pred.reshape(-1, 1))[0][0]
    
    # Update other features based on the last known values
    for col in new_row.columns:
        if col != 'Close':
            new_row[col] = current_data[col].iloc[-1]
    
    current_data = pd.concat([current_data, new_row])
    current_data = current_data.iloc[1:]  # Remove oldest row to maintain same size
    
    # Store first prediction
    future_predictions.append(new_row['Close'].iloc[0])
    future_lower_bounds.append(
        scaler_target.inverse_transform(last_pis[:, 0, 0].reshape(-1, 1))[0][0]
    )
    future_upper_bounds.append(
        scaler_target.inverse_transform(last_pis[:, 1, 0].reshape(-1, 1))[0][0]
    )
    
    # Generate dates for forecasts
    forecast_dates = pd.date_range(
        start=new_row.index[0],
        periods=h_steps,
        freq='D'
    )
    
    # Convert model to evaluation mode
    model.eval()
    
    # Continue with remaining steps
    for step in range(1, h_steps):
        # Preprocess current data
        features, _, _, _ = preprocess_data(
            current_data, 
            delay, 
            embedding_dim, 
            scaler_features=scaler_features,
            scaler_target=scaler_target
        )
        
        # Get last row of features for prediction
        last_features = features[-1:]
        
        # Generate prediction and intervals using MAPIE
        with torch.no_grad():
            y_pred, y_pis = mapie_model.predict(
                last_features,
                alpha=alpha,
                ensemble=True
            )
            
        # Inverse transform predictions and intervals
        pred_value = scaler_target.inverse_transform(y_pred.reshape(-1, 1))
        lower_bound = scaler_target.inverse_transform(y_pis[:, 0, 0].reshape(-1, 1))
        upper_bound = scaler_target.inverse_transform(y_pis[:, 1, 0].reshape(-1, 1))
        
        # Store results
        future_predictions.append(pred_value[0][0])
        future_lower_bounds.append(lower_bound[0][0])
        future_upper_bounds.append(upper_bound[0][0])
        
        # Update current_data for next iteration
        new_row = current_data.iloc[-1:].copy()
        new_row.index = new_row.index + pd.Timedelta(days=1)
        new_row['Close'] = pred_value[0][0]
        
        # Update other features
        for col in new_row.columns:
            if col != 'Close':
                new_row[col] = current_data[col].iloc[-1]
        
        current_data = pd.concat([current_data, new_row])
        # Remove oldest row to maintain same size
        current_data = current_data.iloc[1:]
    return (np.array(future_predictions), 
            np.array(future_lower_bounds), 
            np.array(future_upper_bounds),
            forecast_dates)

class MappieWrapper:
    """Wrapper class to make Neural ODE compatible with scikit-learn API"""
    def __init__(self, model=None): 
        self.model = model
        
    def fit(self, X, y):
        return self
        
    def predict(self, X):
        X_tensor = torch.from_numpy(X).float()
        with torch.no_grad():
            predictions = self.model(X_tensor).numpy()
        return predictions.flatten()

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {"model": self.model}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self