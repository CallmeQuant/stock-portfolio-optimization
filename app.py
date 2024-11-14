import base64
import numpy as np 
import datetime as dt
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import streamlit_shadcn_ui as ui  # type: ignore
from PIL import Image
from interpretations import metric_info, var_info, optimization_strategies_info, \
    appinfo, model_parameters_info, forecast_metrics_info
from portfolio_optimizer import PortfolioOptimizer
from metrics import MetricsCalculator
from risk import RiskMetrics

# Generate market scenario 
from utils import load_config_ml_col, to_numpy, load_and_initialize_model
from data.sp500 import reconstruct_sequences, prepare_sp500_sequences
from Generative.FourierFlow import FourierFlow
from Forecasting.NODE import forecast_stock_prices
from networks.generator import ConditionalLSTMGenerator
from sklearn.metrics import mean_absolute_error, mean_squared_error

PATH_TO_CONFIG = ".\\configs\\config_timegan.yaml"
PATH_TO_MODEL = ".\\checkpoint\\model_dict.pt"

def get_image_base64(image_path):
    """
    Reads an image from the specified path and encodes it to a Base64 string.
    
    Parameters:
    - image_path (str): Path to the image file.
    
    Returns:
    - str: Base64-encoded string of the image.
    """
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def main():
    # Load the page icon
    im = Image.open("EfficientFrontier.png")
    st.set_page_config(page_title="Portfolio Optimization Dashboard", page_icon=im)

    # Sidebar for Application Mode Selection
    st.sidebar.markdown("## Application Mode")
    option = st.sidebar.selectbox(
        "Select Application Mode",
        ("Portfolio Optimization", "Generate Market Scenario", "Stock Price Forecast")
    )
    if option == "Portfolio Optimization":
        # Dashboard Header
        st.markdown("## Portfolio Optimization Dashboard")
        col1, col2 = st.columns([0.14, 0.86], gap="small")
        col1.write("`Project by:`")
        _url = "https://callmequant.github.io/"
        image_base64 = get_image_base64("github.png")
        github_link_html = f'''
                <a href="{_url}" target="_blank" style="text-decoration: none; color: inherit;">
                    <img src="data:image/png;base64,{image_base64}" width="15" height="15" style="vertical-align: middle; margin-right: 10px;">
                    Binh Ho
                </a>
            '''
        col2.markdown(github_link_html, unsafe_allow_html=True)
        
        # App Information
        st.write("""
            **Welcome to the Portfolio Optimizer!**
                 """)
        appinfo()

        # Optimization Strategies
        with st.expander("View Optimization Strategies"):
            optimization_strategies_info()

        # Initialize session state variables
        if "default_stocks" not in st.session_state:
            st.session_state.default_stocks = ["AAPL", "NVDA", "MSFT", "AMZN", "META"]  # Default tickers
            # ['GOOG', 'AAPL', 'FB', 'BABA', 'AMZN', 'GE', 'AMD', 'WMT', 'BAC', 'GM',
            #  'T', 'UAA', 'SHLD', 'XOM', 'RRC', 'BBY', 'MA', 'PFE', 'JPM', 'SBUX']
        if "csv_tickers" not in st.session_state:
            st.session_state.csv_tickers = []
        if "filtered_tickers" not in st.session_state:
            st.session_state.filtered_tickers = []
        if "data_df" not in st.session_state:
            st.session_state.data_df = None

        # Input Parameters Container
        cont1 = st.container()
        cont1.markdown("### Input Parameters")
        
        # File Uploader
        uploaded_file = cont1.file_uploader(
            "Upload a CSV file with Historical Data (Date, Ticker, Adj Close)",
            type=["csv"],
            help="Upload a CSV file containing historical data with columns: Date, Ticker, Adj Close."
        )

        # Process uploaded file
        if uploaded_file is not None:
            try:
                data_df = pd.read_csv(uploaded_file)
                # Validate required columns
                required_columns = {'Date', 'Ticker', 'Adj Close'}
                if not required_columns.issubset(data_df.columns):
                    st.error(f"CSV file must contain the following columns: {', '.join(required_columns)}")
                    st.session_state.csv_tickers = []
                    st.session_state.data_df = None
                else:
                    # Ensure correct data types
                    data_df['Date'] = pd.to_datetime(data_df['Date'])
                    data_df['Ticker'] = data_df['Ticker'].astype(str).str.upper().str.strip()
                    data_df['Adj Close'] = pd.to_numeric(data_df['Adj Close'], errors='coerce')
                    data_df = data_df.dropna(subset=['Date', 'Ticker', 'Adj Close'])
                    st.session_state.data_df = data_df
                    # Extract unique tickers from CSV
                    uploaded_tickers = data_df['Ticker'].unique().tolist()
                    st.session_state.csv_tickers = uploaded_tickers
                    st.success(f"Successfully uploaded data for {len(uploaded_tickers)} tickers.")
            except Exception as e:
                st.error(f"Error reading the uploaded file: {e}")
                st.session_state.csv_tickers = []
                st.session_state.data_df = None
        else:
            st.session_state.csv_tickers = []
            st.session_state.data_df = None

        # Determine the default value for the text input
        if uploaded_file is None:
            # No CSV uploaded: pre-fill with default tickers
            default_tickers_str = ", ".join(st.session_state.default_stocks)
        else:
            # CSV uploaded: leave text input empty to allow users to filter
            default_tickers_str = ""
        
        # Text Input for Tickers
        stocks = cont1.text_input(
            "Enter Tickers to Use (separated by commas)",
            value=default_tickers_str,
            help=(
                "If a CSV file is uploaded, enter tickers to filter the CSV. "
                "Leave empty to use all tickers from the CSV. "
                "If no CSV is uploaded, enter tickers separated by commas. "
                "Default tickers are used only when no CSV is uploaded and no input is provided."
            )
        )
        
        # Date Inputs
        start, end = cont1.columns(2)
        if uploaded_file is not None and st.session_state.data_df is not None:
            min_date = st.session_state.data_df['Date'].min().date()
            max_date = st.session_state.data_df['Date'].max().date()
        else:
            min_date = dt.date.today() - dt.timedelta(days=365*5)
            max_date = dt.date.today()
        
        start_date = start.date_input(
            "Start Date",
            max_value=max_date - dt.timedelta(days=1),
            min_value=min_date,
            value=dt.date.today() - dt.timedelta(days=365),
        )
        end_date = end.date_input(
            "End Date",
            max_value=max_date,
            min_value=start_date + dt.timedelta(days=1),
            value=dt.date.today(),
        )
        
        # Optimization Criterion and Risk-Free Rate
        col1, col2 = cont1.columns(2)
        optimization_criterion = col1.selectbox(
            "Optimization Objective",
            options=[
                "Maximize Sharpe Ratio",
                "Minimize Volatility",
                "Maximize Sortino Ratio",
                "Minimize Tracking Error",
                "Maximize Information Ratio",
                "Minimize Conditional Value-at-Risk",
                "Hierarchical Risk Parity", 
                "Hierarchical Equal Risk Contribution",
                "Distributionally Robust CVaR",
                "Black Litterman Empirical",
                "Minimize EVaR",
                "Risk Parity EDaR"
            ],
        )
        riskFreeRate_d = col2.number_input(
            "Risk Free Rate (%)",
            min_value=0.00,
            max_value=100.00,
            step=0.001,
            format="%0.3f",
            value=3.791,
            help="10 Year Bond Yield"
        )
        calc = cont1.button("Construct portfolio")
        riskFreeRate = riskFreeRate_d / 100

        # Initialize stocks list based on CSV upload and text input
        if uploaded_file is not None and st.session_state.csv_tickers and st.session_state.data_df is not None:
            if stocks.strip():
                # Use text input to filter CSV tickers
                input_tickers = [s.strip().upper() for s in stocks.split(",") if s.strip()]
                # Filter CSV tickers based on input
                filtered_tickers = [ticker for ticker in input_tickers if ticker in st.session_state.csv_tickers]
                missing_tickers = set(input_tickers) - set(filtered_tickers)
                if missing_tickers:
                    st.warning(f"The following tickers from your input were not found in the uploaded CSV and will be ignored: {', '.join(missing_tickers)}")
                st.session_state.filtered_tickers = filtered_tickers
            else:
                # Use all tickers from CSV
                st.session_state.filtered_tickers = st.session_state.csv_tickers.copy()
            
            st.session_state.stocks_list = st.session_state.filtered_tickers
        else:
            # No CSV uploaded: use text input or default tickers
            if stocks.strip():
                st.session_state.stocks_list = [s.strip().upper() for s in stocks.split(",") if s.strip()]
            else:
                # Use default tickers if no input is provided
                st.session_state.stocks_list = st.session_state.default_stocks.copy()

        if calc:
            if not st.session_state.stocks_list:
                st.error("No tickers available. Please enter tickers or upload a CSV file.")
                return

            try:
                with st.spinner("Buckle Up! Road to heaven in Progress...."):
                    stocks_list = st.session_state.stocks_list
                    
                    if uploaded_file is not None and st.session_state.data_df is not None:
                        # Use data from uploaded CSV
                        optimizer = PortfolioOptimizer(
                            stocks=stocks_list,
                            start=start_date.strftime('%Y-%m-%d'),
                            end=end_date.strftime('%Y-%m-%d'),
                            optimization_criterion=optimization_criterion,
                            riskFreeRate=riskFreeRate,
                            data_df=st.session_state.data_df
                        )
                    else:
                        # Use yfinance to download data
                        optimizer = PortfolioOptimizer(
                            stocks=stocks_list,
                            start=start_date.strftime('%Y-%m-%d'),
                            end=end_date.strftime('%Y-%m-%d'),
                            optimization_criterion=optimization_criterion,
                            riskFreeRate=riskFreeRate
                        )
                    
                    ret, std = optimizer.basicMetrics()
                    if not (len(ret.columns) == len(stocks_list)):
                        missing_tickers = set(stocks_list) - set(ret.columns)
                        raise ValueError(
                            f"Data for the following tickers could not be retrieved: {', '.join(missing_tickers)}"
                        )

                    optimizer.optimized_allocation.columns = ["Allocation (%)"]
                    optimizer.optimized_allocation["Allocation (%)"] = [
                        round(i * 100, 2)
                        for i in optimizer.optimized_allocation["Allocation (%)"]
                    ]

                    metrics = MetricsCalculator(
                        stocks_list,
                        start_date,
                        end_date,
                        optimization_criterion,
                        riskFreeRate,
                    )
                    
                    metric_df = metrics.metricDf()
                    metric_df = pd.DataFrame(list(metric_df.items()))
                    metric_df.columns = ["Metric", "Value"]

                    riskM = RiskMetrics(
                        stocks_list,
                        start_date,
                        end_date,
                        optimization_criterion,
                        riskFreeRate,
                    )

            except ValueError as e:
                st.error("Unable to download data for one or more tickers!")
                st.error(str(e))
                return
            except Exception as e:
                st.error(str(e))
                return

            # Display Results in Tabs
            with st.container():
                tab1, tab2, tab3, tab4, tab5 = st.tabs(
                    [
                        "Summary",
                        "Efficient Frontier",
                        "Metrics",
                        "Portfolio Returns",
                        "Risk Analysis",
                    ]
                )
                with tab1:
                    st.markdown("#### Optimized Portfolio Performance")
                    col1, col2 = st.columns(2)
                    col1.markdown(f"**Returns**: {optimizer.optimized_returns}%")
                    col1.markdown(f"**Volatility**: {optimizer.optimized_std}%")
                    sharpe = (
                        optimizer.optimized_returns - (optimizer.riskFreeRate * 100)
                    ) / optimizer.optimized_std
                    col1.markdown(f"**Sharpe Ratio**: {round(sharpe, 2)}")
                    col1.markdown(f"**Sortino Ratio**: {round(metrics.MSortinoRatio(), 2)}")
                    col2.markdown(f"**Time Period**: {(end_date - start_date).days} days")
                    st.markdown("#### Optimized Portfolio Allocation")
                    alocCol, pieCol = st.columns(2)
                    with alocCol:
                        allocations = optimizer.optimized_allocation
                        allocations["Tickers"] = allocations.index
                        allocations = allocations[["Tickers", "Allocation (%)"]]
                        ui.table(allocations)
                    with pieCol:
                        sharpeChart = optimizer.optimized_allocation[
                            optimizer.optimized_allocation["Allocation (%)"] != 0
                        ]
                        fig = px.pie(
                            sharpeChart, values="Allocation (%)", names=sharpeChart.index
                        )
                        fig.update_layout(
                            width=180,
                            height=200,
                            showlegend=False,
                            margin=dict(t=20, b=0, l=0, r=0),
                        )
                        st.plotly_chart(fig, use_container_width=True)

                with tab2:
                    st.markdown("#### Efficient Frontier Assets")
                    frontierAssets, matrix = optimizer.frontierStats()
                    ui.table(frontierAssets)
                    st.markdown("#### Asset Correlations")
                    ui.table(matrix)
                    st.markdown("*(Higher Value Represents Higher Correlation)*")
                    st.markdown("#### Efficient Frontier Graph")
                    optimizer.EF_graph()

                with tab3:
                    st.markdown("#### Risk and Return Metrics")
                    ui.table(metric_df)
                    with st.expander("Metric Interpretations:"):
                        metric_info()

                with tab4:
                    st.markdown("#### Cumulative Portfolio Returns")
                    metrics.portfolioReturnsGraph()

                with tab5:
                    st.markdown("#### VaR and CVaR")
                    var = riskM.riskTable()
                    ui.table(var)
                    with st.expander("VaR and CVar Interpretation"):
                        var_info()
                    st.markdown("#### VaR Breaches")
                    riskM.varXReturns()
                    
    elif option == "Generate Market Scenario":
        st.markdown("## Generate Market Scenario")
        st.write("""
            **Welcome to the Market Scenario Generator!**

            Dive into the future by simulating potential market movements of the S&P 500 index. Our sophisticated pre-trained GAN model harnesses historical data to create realistic market scenarios. This tool is perfect for stress-testing your investment strategies against a variety of market conditions.

            *Specify the number of days and click the button below to generate a new market scenario and explore possible futures!*
        """)
        # Input fields for hyperparameters
        st.subheader("Model Specification (use for re-train only)")
        n_lags = st.number_input("Number of lookback days:", min_value=1, value=20, step=1)
        hidden_dim = st.number_input("Hidden Dimension:", min_value=1, value=50, step=1)
        n_flows = st.number_input("Number of Flows:", min_value=1, value=20, step=1)
        st.info("Note: It's recommended to set number of lookback periods equal to number of flows.")

        with st.expander("View Model Parameters"):
        # Explanation for Hidden Dimension and Number of Flows
            model_parameters_info()
        
        # Input for start date
        start_date = st.date_input(
            "Select the start date for training data (use for re-train only):",
            value=dt.date.today() - dt.timedelta(days=365*5),
            min_value=dt.date(2000, 1, 1),
            max_value=dt.date.today() - dt.timedelta(days=1)
        )

        # Checkbox to retrain the model
        retrain_model = st.checkbox("Retrain model with specified parameters")

        # Input for number of days
        max_days = n_lags
        if retrain_model:
            num_days = st.number_input(
                "Enter the number of days to generate (max {} days):".format(max_days),
                min_value=5,
                max_value=max_days,
                value=5,
                step=1
            )
        else:
            num_days = st.number_input(
                "Enter the number of days to generate:",
                min_value=5,
                value=5,
                step=1
            )

        # Button to generate the market scenario
        if st.button("Generate Market Scenario"):
            if retrain_model:
                # st.write("Retraining the model with specified choices...")
                with st.spinner("Retraining the model with specified choices..."):
                    # Prepare training data
                    training_data, _, sp500_df = prepare_sp500_sequences(start=start_date,
                                                                        sequence_length=n_lags, 
                                                                        train_test_split=1.)
                    model = FourierFlow(
                        input_dim=1,
                        output_dim=1,
                        hidden=hidden_dim,
                        n_flows=n_flows,
                        n_lags=n_lags,
                        FFT=True,
                        flip=True,
                        normalize=False)
                    model.fit(training_data, epochs=30, batch_size=128, display_step=10)
            else:
                config = load_config_ml_col(PATH_TO_CONFIG)
                _, _, sp500_df = prepare_sp500_sequences(train_test_split=1.)
                model = ConditionalLSTMGenerator(
                            input_dim=config.G_input_dim, 
                            hidden_dim=config.G_hidden_dim, 
                            output_dim=config.input_dim,
                            n_layers=config.G_num_layers
                        )
                model = load_and_initialize_model(model, PATH_TO_MODEL, device='cpu')
                model.eval()

            # Generate samples
            num_samples = 10  # Number of samples to generate
            if retrain_model:
                generated_returns = to_numpy(model.sample(num_samples))  # Shape: (num_samples, n_lags, 1)
            else:
                generated_returns = to_numpy(model(num_samples, num_days, device=config.device)) # Shape: (num_samples, num_days, 1)
            
            sequence_length = generated_returns.shape[1]

            # Reconstruct sequences based on retrain_model condition
            if retrain_model:
                generated_returns_re = reconstruct_sequences(generated_returns, step=sequence_length)
                if num_days <= sequence_length:
                    # Use a single sample and take the first num_days time steps
                    selected_sample = generated_returns[:, :num_days, :]  # Shape: (num_days, 1)
                else:
                    st.warning("Requested number of days exceeds the maximum available length of generated data. Reducing number of days to {}.".format(max_days))
                    num_days = max_days
                    selected_sample = generated_returns_re[:, :num_days]
            else:
                selected_sample = generated_returns.copy()

            last_date_price = sp500_df['Adj Close'].iloc[-1]
            simulated_dates = pd.date_range(start=dt.date.today(), periods=num_days, freq='B')
            # Calculate the simulated prices
            simulated_prices = np.exp(np.cumsum(selected_sample, axis=1)) * last_date_price

            # Create a DataFrame for the simulated prices
            simulated_df = pd.DataFrame({'Date': simulated_dates})
            for i in range(num_samples):
                simulated_df[f'Sample {i+1}'] = simulated_prices[i, :]
            simulated_df.set_index('Date', inplace=True)
            st.success("Market scenario generated successfully.")
            
            # Reshape generated returns for plotting
            reshaped_returns = selected_sample.squeeze(-1)  # Shape: (num_samples, num_days)

            # Plot Generated Returns
            returns_df = pd.DataFrame(reshaped_returns.T, columns=[f'Sample {i+1}' for i in range(num_samples)])
            returns_df['Date'] = simulated_dates
            returns_df.set_index('Date', inplace=True)
            
            fig_returns = px.line(
                            returns_df,
                            x=returns_df.index,
                            y=returns_df.columns,
                            title='Generated Returns',
                            labels={'variable': 'Sample', 'value': 'Return'},
                        )
            fig_returns.update_layout(
                xaxis_title='Date',
                yaxis_title='Generated S&P 500 Return',
                margin=dict(t=50, b=0, l=0, r=0),
            )
            st.plotly_chart(fig_returns, use_container_width=False)
            
            fig = px.line(
                        simulated_df,
                        x=simulated_df.index,
                        y=simulated_df.columns,
                        title='Simulated S&P 500 Index',
                        labels={'variable': 'Sample', 'value': 'Price'},
                    )
            fig.update_layout(
                xaxis_title='Date',
                yaxis_title='Simulated S&P 500 Price',
                margin=dict(t=50, b=0, l=0, r=0),
            )
            st.plotly_chart(fig, use_container_width=False)

    elif option == "Stock Price Forecast": 
        st.markdown("## Stock Price Forecast")
        st.write("""
            **Welcome to the Stock Price Forecaster!**

            Predict the future prices of your favorite stocks with our cutting-edge forecasting tools. Whether you're interested in a single stock or a portfolio, our lightweight yet powerful models provide quick and accurate forecasts to inform your investment decisions.

            *Enter the stock symbols below to start forecasting and gain insights into future price movements!*
        """)

        # Select forecast type
        # forecast_type = st.radio("Select Forecast Type", ("Univariate (Single Stock)", "Multivariate (Multiple Stocks)"))

        # Input for stock symbols
        # stock_symbols = st.text_input(
        #     "Enter Stock Symbol" if forecast_type == "Univariate (Single Stock)" else "Enter Stock Symbols (separated by commas)",
        #     value="AAPL" if forecast_type == "Univariate (Single Stock)" else "AAPL, MSFT"
        # )
        stock_symbols = st.text_input(
            "Enter Stock Symbol",value="AAPL")

        # Convert input to list
        stock_list = [s.strip().upper() for s in stock_symbols.split(",") if s.strip()]

        # Date inputs for training period
        col1, col2 = st.columns(2)
        start_date = col1.date_input(
            "Training Start Date",
            value=dt.date.today() - dt.timedelta(days=365*2),
            max_value=dt.date.today() - dt.timedelta(days=1)
        )
        end_date = col2.date_input(
            "Training End Date",
            value=dt.date.today(),
            min_value=start_date + dt.timedelta(days=1),
            max_value=dt.date.today()
        )
        
        # Confidence level input
        confidence_level = st.slider(
            "Confidence Level (%)", 
            min_value=85,
            max_value=99,
            value=90,
            step=1,
            help="Select the confidence level for prediction intervals"
        )
        # Partial fit 
        partial_fit = st.checkbox("Full forecast intervals",
                                  help = "Useful to capture uncertainty caused by sudden shifts!")
        dates = pd.date_range(start=start_date, end=end_date)
        test_data_length = len(dates[:int(len(dates)* 0.8)])
        # Add these input parameters in the sidebar or main area
        col1, col2 = st.columns(2)
        with col1:
            forecast_period = st.number_input(
                "Forecast Period (days)", 
                min_value=1, 
                max_value=30,  # You can adjust this maximum
                value=5,
                help="Number of days to forecast ahead. Note: Longer forecast periods may lead to constant predictions due to model limitations."
            )

        with col2:
            context_window = st.number_input(
                "Context Window Size",
                min_value=5,
                max_value=test_data_length,
                value=min(50, test_data_length),
                help="Number of previous days used for generating forecasts. Should be between 5 and the length of test data."
            )

        # Button to start forecast
        if st.button("Run Stock Price Forecast"):
            if not stock_list:
                st.error("Please enter one stock symbol.")
            else:
                with st.spinner("Running stock price forecast..."):
                    alpha = 1 - (confidence_level / 100)
                    predicted_prices, actual_prices, \
                        lower_bound, upper_bound, test_dates,\
                        coverage, width, cwc, \
                        future_pred, future_lower, future_upper, future_dates, \
                             train_data, test_data = forecast_stock_prices(
                            stock_symbols.upper(),
                            start_date,
                            end_date,
                            h_steps=forecast_period,
                            context_window=context_window,
                            alpha=alpha,
                            partial_fit=partial_fit
                        )
                    
                    if all(x is not None for x in [predicted_prices, actual_prices, 
                                                lower_bound, upper_bound, test_dates,
                                                coverage, width, cwc]):
                        # Calculate metrics
                        mae = mean_absolute_error(actual_prices, predicted_prices)
                        rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
                        
                        # Display metrics
                        col1, col2, col3, col4, col5 = st.columns(5)
                        col1.metric("MAE", f"{mae:.2f}")
                        col2.metric("RMSE", f"{rmse:.2f}")
                        col3.metric("Coverage Score", f"{coverage:.2f}")
                        col4.metric("Width Score", f"{width:.2f}")
                        col5.metric("Coverage-Width", f"{cwc:.2f}")
                        
                        with st.expander("View Forecast Metrics"):
                            forecast_metrics_info()

                        # First chart: Full visualization
                        st.subheader("Full Training, Testing, and Forecast Visualization")
                        fig1 = go.Figure()
                        
                        # Add training data
                        fig1.add_trace(go.Scatter(
                            x=train_data.index,
                            y=train_data['Close'],
                            name="Training Data",
                            line=dict(color="darkgray", width=1)
                        ))
                        
                        # Add actual test prices
                        fig1.add_trace(go.Scatter(
                            x=test_dates[-len(actual_prices):],
                            y=actual_prices.flatten(),
                            name="Actual",
                            line=dict(color="blue")
                        ))
                        
                        # Add predicted prices
                        fig1.add_trace(go.Scatter(
                            x=test_dates[-len(actual_prices):],
                            y=predicted_prices.flatten(),
                            name="Predicted",
                            line=dict(color="orange")
                        ))
                        
                        # Add in-sample prediction intervals
                        fig1.add_trace(go.Scatter(
                            x=test_dates[-len(actual_prices):].tolist() + 
                                test_dates[-len(actual_prices):].tolist()[::-1],
                            y=np.concatenate([upper_bound.flatten(), lower_bound.flatten()[::-1]]),
                            fill='toself',
                            fillcolor='rgba(255,165,0,0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name=f'In-sample {confidence_level}% PI'
                        ))
                        
                        # Add future forecast
                        fig1.add_trace(go.Scatter(
                            x=future_dates,
                            y=future_pred,
                            name="Forecast",
                            line=dict(color="red", dash='dash')
                        ))
                        
                        # Add out-of-sample prediction intervals
                        fig1.add_trace(go.Scatter(
                            x=future_dates.tolist() + future_dates.tolist()[::-1],
                            y=np.concatenate([future_upper, future_lower[::-1]]),
                            fill='toself',
                            fillcolor='rgba(255,0,0,0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name=f'Out-of-sample {confidence_level}% PI'
                        ))
                        
                        # Add vertical lines for period separation
                        fig1.add_vline(x=train_data.index[-1], line_dash="dash", line_color="gray")
                        fig1.add_vline(x=test_dates[-1], line_dash="dash", line_color="gray")
                        
                        fig1.update_layout(
                            title=f'{stock_symbols} Complete Stock Price Analysis',
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            showlegend=True,
                            margin=dict(t=50, b=0, l=0, r=0),
                        )
                        
                        st.plotly_chart(fig1, use_container_width=True)
                        
                        # Second chart: Test and Forecast only
                        st.subheader("Test Period and Forecast Visualization")
                        fig2 = go.Figure()
                        
                        # Add actual test prices
                        fig2.add_trace(go.Scatter(
                            x=test_dates[-len(actual_prices):],
                            y=actual_prices.flatten(),
                            name="Actual",
                            line=dict(color="blue")
                        ))
                        
                        # Add predicted prices
                        fig2.add_trace(go.Scatter(
                            x=test_dates[-len(actual_prices):],
                            y=predicted_prices.flatten(),
                            name="Predicted",
                            line=dict(color="orange")
                        ))
                        
                        # Add in-sample prediction intervals
                        fig2.add_trace(go.Scatter(
                            x=test_dates[-len(actual_prices):].tolist() + 
                                test_dates[-len(actual_prices):].tolist()[::-1],
                            y=np.concatenate([upper_bound.flatten(), lower_bound.flatten()[::-1]]),
                            fill='toself',
                            fillcolor='rgba(255,165,0,0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name=f'In-sample {confidence_level}% PI'
                        ))
                        
                        # Add future forecast
                        fig2.add_trace(go.Scatter(
                            x=future_dates,
                            y=future_pred,
                            name="Forecast",
                            line=dict(color="red", dash='dash')
                        ))
                        
                        # Add out-of-sample prediction intervals
                        fig2.add_trace(go.Scatter(
                            x=future_dates.tolist() + future_dates.tolist()[::-1],
                            y=np.concatenate([future_upper, future_lower[::-1]]),
                            fill='toself',
                            fillcolor='rgba(255,0,0,0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name=f'Out-of-sample {confidence_level}% PI'
                        ))
                        
                        fig2.update_layout(
                            title=f'{stock_symbols} Test and Forecast Period Analysis',
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            showlegend=True,
                            margin=dict(t=50, b=0, l=0, r=0),
                        )
                        
                        st.plotly_chart(fig2, use_container_width=True)
                        
                        # Display forecast table
                        st.subheader("Out-of-sample Forecast Results")
                        forecast_df = pd.DataFrame({
                            'Date': future_dates,
                            'Forecast': future_pred,
                            f'Lower Bound ({confidence_level}%)': future_lower,
                            f'Upper Bound ({confidence_level}%)': future_upper
                        })
                        
                        # Format the numbers in the DataFrame
                        for col in forecast_df.columns:
                            if col != 'Date':
                                forecast_df[col] = forecast_df[col].round(2)
                        
                        # Display the table with scrolling
                        st.dataframe(
                            forecast_df.style.format({
                                'Date': lambda x: x.strftime('%Y-%m-%d'),
                                'Forecast': '{:.2f}',
                                f'Lower Bound ({confidence_level}%)': '{:.2f}',
                                f'Upper Bound ({confidence_level}%)': '{:.2f}'
                            }),
                            height=200  # Adjust this value based on your needs
                        )
                        st.success("Forecast completed successfully!")
        
if __name__ == "__main__":
    main()

