import yfinance as yf
import numpy as np
import scipy.optimize as sc
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime, timedelta
from scipy.stats import norm
from sqlalchemy import create_engine
from google.cloud import storage
import io
import warnings
from skfolio import RiskMeasure, MultiPeriodPortfolio
from skfolio.optimization import HierarchicalEqualRiskContribution, \
    HierarchicalRiskParity, DistributionallyRobustCVaR, MeanRisk, ObjectiveFunction, RiskBudgeting
from skfolio.prior import EmpiricalPrior
from skfolio.moments import ShrunkMu, GraphicalLassoCV
from skfolio import RiskMeasure
from skfolio.cluster import HierarchicalClustering, LinkageMethod
from skfolio.moments import ShrunkMuMethods

class PortfolioOptimizer:

    def __init__(
        self, stocks, start, end, optimization_criterion="Maximize Sharpe Ratio", 
        riskFreeRate=0.07024,
        data_file = None,
        data_df = None,
        db_connection_str=None,
        db_query=None,
        gcs_bucket_name=None,
        gcs_file_path=None):
        """
        Initializes the PortfolioOptimizer class.

        Args:
            stocks (list): List of stock tickers.
            start (str, optional): Start date in 'YYYY-MM-DD' format.
            end (str, optional): End date in 'YYYY-MM-DD' format.
            optimization_criterion (str): Criterion for optimization. Defaults to 'Maximize Sharpe Ratio'.
            riskFreeRate (float): Risk-free rate for calculations. Defaults to 0.07024.
            data_file (str, optional): Path to the CSV file containing data.
            data_df (pd.DataFrame, optional): DataFrame containing the data.
            db_connection_str (str, optional): Database connection string.
            db_query (str, optional): SQL query to fetch data from the database.
        Raises:
            ValueError: If insufficient parameters are provided.
        """
        self.stocks = [stock.upper() for stock in stocks]
        self.start = start
        self.end = end
        self.optimization_criterion = optimization_criterion
        self.riskFreeRate = riskFreeRate
        
        # Initialize data source flags
        self.use_csv_data = data_file is not None
        self.use_db_data = db_connection_str is not None and db_query is not None
        self.use_df_data = data_df is not None
        self.use_gcs_data = gcs_bucket_name is not None and gcs_file_path is not None

        # Load data based on the provided source
        if self.use_csv_data:
            self.data = self.load_data_from_file(data_file)
        elif self.use_db_data:
            self.data = self.load_data_from_db(db_connection_str, db_query)
        elif self.use_df_data:
            self.data = self.load_data_from_df(data_df)
        else:
            # If no data source is provided, use yfinance (existing behavior)
            self.data = None
            if self.start is None or self.end is None:
                warnings.warn("Start and end dates are not provided. Using 1 year from now as date range.")
                self.end = datetime.now().strftime('%Y-%m-%d')
                self.start = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

        # Initialize other attributes
        self.meanReturns, self.covMatrix = self.getData()
        self.benchmark = self.benchmarkReturns()
        (
            self.optimized_returns,
            self.optimized_std,
            self.optimized_allocation,
            self.efficientList,
            self.targetReturns,
        ) = self.calculatedResults()

    def load_data_from_file(self, data_file):
        """
        Loads data from the provided CSV file and prepares it for analysis.

        Returns:
            pd.DataFrame: A DataFrame with Date as index, tickers as columns, and return_t as values.
        """
        final_data = pd.read_csv(data_file)
        return self.process_data(final_data)

    def load_data_from_db(self, connection_str, query):
        """
        Loads data from the database using the provided connection string and query.

        Args:
            connection_str (str): Database connection string.
            query (str): SQL query to fetch data.

        Returns:
            pd.DataFrame: Processed data ready for analysis.
        """
        engine = create_engine(connection_str)
        final_data = pd.read_sql_query(query, engine)
        return self.process_data(final_data)
    
    def load_data_from_df(self, data_df):
        """
        Processes the provided DataFrame for analysis.

        Args:
            data_df (pd.DataFrame): DataFrame containing the data.

        Returns:
            pd.DataFrame: Processed data ready for analysis.
        """
        return self.process_data(data_df)
    
    def load_data_from_gcs(self, bucket_name, file_path):
        """
        Loads data from a Google Cloud Storage bucket and processes it.

        Args:
            bucket_name (str): Name of the GCS bucket.
            file_path (str): Path to the file within the bucket.

        Returns:
            pd.DataFrame: Processed data ready for analysis.
        """
        # Create a client
        client = storage.Client()

        # Get the bucket
        bucket = client.get_bucket(bucket_name)

        # Get the blob
        blob = bucket.blob(file_path)

        # Download the blob's content as bytes
        data_bytes = blob.download_as_bytes()

        # Read the data into a pandas DataFrame
        final_data = pd.read_csv(io.BytesIO(data_bytes))

        return self.process_data(final_data)
    
    def process_data(self, final_data):
        """
        Processes the raw data into the format required for analysis.

        Args:
            final_data (pd.DataFrame): Raw data.

        Returns:
            pd.DataFrame: Processed data.
        """
        final_data = final_data[['Date', 'Ticker', 'Adj Close']]
        # Pivot wider
        final_data_pivot = final_data.pivot(index='Date', columns='Ticker', values='Adj Close')
        final_data_pivot.index = pd.to_datetime(final_data_pivot.index)
        # Filter the data for the specified date range if provided
        if self.start is not None and self.end is not None:
            if self.start < final_data_pivot.index[0].strftime("%Y-%m-%d") or self.end > final_data_pivot.index[-1].strftime("%Y-%m-%d"):
                warnings.warn("The provided start date and end date must be included in the date range of the data")
                self.start = final_data_pivot.index[0].strftime("%Y-%m-%d")
                self.end = final_data_pivot.index[-1].strftime("%Y-%m-%d")
            final_data_pivot = final_data_pivot.loc[self.start:self.end]
        # only selected stocks
        final_data_pivot = final_data_pivot.loc[:, self.stocks]
        final_data_pivot = final_data_pivot.dropna()
        return final_data_pivot 
    
    def basicMetrics(self):
        if not all(s.isupper() for s in self.stocks):
            raise ValueError("Enter ticker names in Capital Letters!")
        if len(self.stocks) <= 1:
            raise ValueError("More than 1 ticker input required!")
        if self.use_csv_data or self.use_db_data or self.use_df_data:
            returns = self.data
            returns = returns[self.stocks]  # Double check step
            stdIndividual = returns.std()
            return returns, stdIndividual
        else:
            try:
                stockData = yf.download(self.stocks, start=self.start, end=self.end)
            except:
                raise ValueError("Unable to download data, try again later!")
            stockData = stockData["Adj Close"]

            if len(stockData.columns) != len(self.stocks):
                raise ValueError("Unable to download data for one or more tickers!")

            # returns = stockData.pct_change()
            returns = np.log(stockData / stockData.shift(1))
            stdIndividual = returns.std()

            return returns, stdIndividual

    def portfolioReturnsDaily(self):
        dailyIndividualReturns, dailyIndividualStd = self.basicMetrics()
        portfolioDailyReturns = np.dot(
            dailyIndividualReturns.dropna(), self.optimized_allocation
        )
        return portfolioDailyReturns

    def benchmarkReturns(self):
        try:
            benchmark_data = yf.download("^SPX", self.start, self.end)
        except:
            raise ValueError("Unable to download data, try again later!")
        # benchmark_returns = benchmark_data["Adj Close"].pct_change().dropna()
        benchmark_returns = np.log(benchmark_data["Adj Close"] / benchmark_data["Adj Close"].shift(1)).dropna()
        return benchmark_returns

    def getData(self):

        returns, stdIndividual = self.basicMetrics()
        meanReturns = (returns.mean())  
        covMatrix = (returns.cov())  

        return meanReturns, covMatrix

    def portfolioPerformance(self, weights):
        returns = (np.sum(self.meanReturns * weights) * 252)  
        std = np.sqrt(np.dot(weights.T, np.dot(self.covMatrix, weights))) * np.sqrt(252)  
        return returns, std

    def sharpe(self, weights):
        pReturns, pStd = self.portfolioPerformance(weights)  
        return (-(pReturns - self.riskFreeRate) / pStd)  

    def sortino(self, weights):
        dailyIndividualReturns, dailyIndividualStd = self.basicMetrics()
        portfolioDailyReturns = np.dot(dailyIndividualReturns.dropna(), weights)
        downsideChanges = portfolioDailyReturns[portfolioDailyReturns < 0]
        downside_deviation = downsideChanges.std(ddof=1) * np.sqrt(252)
        meanReturns = portfolioDailyReturns.mean() * 252
        sortino_ratio = (meanReturns - self.riskFreeRate) / downside_deviation

        return -sortino_ratio

    def portfolioVariance(self, weights):  
        return self.portfolioPerformance(weights)[1]

    def trackingError(self, weights):
        dailyIndividualReturns, dailyIndividualStd = self.basicMetrics()
        portfolioDailyReturns = np.array(np.dot(dailyIndividualReturns.dropna(), weights))
        benchmarkReturns = np.array(self.benchmark)

        difference_array = portfolioDailyReturns - benchmarkReturns
        trackingError = difference_array.std(ddof=1) * np.sqrt(252)

        return trackingError

    def informationRatio(self, weights):
        dailyIndividualReturns, dailyIndividualStd = self.basicMetrics()
        portfolioDailyReturns = np.array(np.dot(dailyIndividualReturns.dropna(), weights))
        benchmarkReturns = np.array(self.benchmark)
        difference_array = portfolioDailyReturns - benchmarkReturns
        portfolioPerformance = portfolioDailyReturns.mean() * 252
        benchmarkPerformance = benchmarkReturns.mean() * 252
        trackingError = difference_array.std(ddof=1) * np.sqrt(252)

        information = (portfolioPerformance - benchmarkPerformance) / trackingError

        return -information

    def conditionalVar(self, weights):
        dailyIndividualReturns, dailyIndividualStd = self.basicMetrics()
        portfolioDailyReturns = np.array(np.dot(dailyIndividualReturns.dropna(), weights))
        mu = portfolioDailyReturns.mean()
        sigma = portfolioDailyReturns.std(ddof=1)
        var = mu + sigma * norm.ppf(0.95)
        loss = portfolioDailyReturns[portfolioDailyReturns < -var]
        cvar = np.mean(loss)

        return -cvar

    def optimization_function(self, constraintSet=(0, 1)):

        numAssets = len(self.meanReturns)  ## gets the number of stocks in the portfolio
        constraints = {
            "type": "eq",
            "fun": lambda x: np.sum(x) - 1,
        }  
        bound = constraintSet  
        bounds = tuple(bound for asset in range(numAssets)) 

        if self.optimization_criterion == "Maximize Sharpe Ratio":
            return sc.minimize(
                self.sharpe,
                numAssets * [1.0 / numAssets],  
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )
        elif self.optimization_criterion == "Minimize Volatility":
            return sc.minimize(
                self.portfolioVariance,
                numAssets * [1.0 / numAssets],  
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )
        elif self.optimization_criterion == "Maximize Sortino Ratio":
            return sc.minimize(
                self.sortino,
                numAssets * [1.0 / numAssets],  
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )
        elif self.optimization_criterion == "Minimize Tracking Error":
            return sc.minimize(
                self.trackingError,
                numAssets * [1.0 / numAssets],  
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )
        elif self.optimization_criterion == "Maximize Information Ratio":
            return sc.minimize(
                self.informationRatio,
                numAssets * [1.0 / numAssets],
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )
        elif self.optimization_criterion == "Minimize Conditional Value-at-Risk":
            return sc.minimize(
                self.conditionalVar,
                numAssets * [1.0 / numAssets],  
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )
        
        elif self.optimization_criterion == "Hierarchical Risk Parity":
            # Use skportfolio
            dailyIndividualReturns, dailyIndividualStd = self.basicMetrics()
            returns_data = dailyIndividualReturns.dropna()

            model = HierarchicalRiskParity(
                risk_measure=RiskMeasure.CVAR,
                hierarchical_clustering_estimator=HierarchicalClustering(
                    linkage_method=LinkageMethod.SINGLE,
                ),
                portfolio_params=dict(name="HRP-CVaR-Single-Pearson"),
            )

            model.fit_predict(returns_data)

            optimized_allocation = model.weights_ 

            return optimized_allocation
        
        elif self.optimization_criterion == "Hierarchical Equal Risk Contribution":
            # Use skportfolio
            dailyIndividualReturns, dailyIndividualStd = self.basicMetrics()
            returns_data = dailyIndividualReturns.dropna()

            model = HierarchicalEqualRiskContribution(
                risk_measure=RiskMeasure.CDAR,
                hierarchical_clustering_estimator=HierarchicalClustering(
                    linkage_method=LinkageMethod.SINGLE,
                ),
                portfolio_params=dict(name="HERC-CVaR-Single-Pearson"),
            )

            model.fit_predict(returns_data)

            optimized_allocation = model.weights_  # pd.Series

            return optimized_allocation
        
        elif self.optimization_criterion == "Distributionally Robust CVaR":
            # Use skportfolio
            dailyIndividualReturns, dailyIndividualStd = self.basicMetrics()
            returns_data = dailyIndividualReturns.dropna()

            model = DistributionallyRobustCVaR(
                    wasserstein_ball_radius=0.001,
                    portfolio_params=dict(name="Distributionally Robust CVaR - 0.001"),
                )
            model.fit_predict(returns_data)
            optimized_allocation = model.weights_  # pd.Series

            return optimized_allocation
        
        elif self.optimization_criterion == "Black Litterman Empirical":
            # Use skportfolio
            dailyIndividualReturns, dailyIndividualStd = self.basicMetrics()
            returns_data = dailyIndividualReturns.dropna()
            model = MeanRisk(
                    risk_measure=RiskMeasure.VARIANCE,
                    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
                    prior_estimator=EmpiricalPrior(
                        mu_estimator=ShrunkMu(covariance_estimator=GraphicalLassoCV(), 
                                              vol_weighted_target=True,
                                              method=ShrunkMuMethods.BODNAR_OKHRIN),
                        covariance_estimator=GraphicalLassoCV()),
                    portfolio_params=dict(name="Empirical"))
            
            model.fit_predict(returns_data)
            optimized_allocation = model.weights_  # pd.Series

            return optimized_allocation
        
        elif self.optimization_criterion == "Minimize EVaR":
            # Use skportfolio
            dailyIndividualReturns, dailyIndividualStd = self.basicMetrics()
            returns_data = dailyIndividualReturns.dropna()
            model = MeanRisk(
                    risk_measure=RiskMeasure.EVAR,
                    objective_function=ObjectiveFunction.MINIMIZE_RISK,
                    prior_estimator=EmpiricalPrior(
                        mu_estimator=ShrunkMu(covariance_estimator=GraphicalLassoCV(), 
                                              vol_weighted_target=True,
                                              method=ShrunkMuMethods.BODNAR_OKHRIN),
                        covariance_estimator=GraphicalLassoCV()),
                    portfolio_params=dict(name="Minimize EVaR"))
            
            model.fit_predict(returns_data)
            optimized_allocation = model.weights_  # pd.Series

            return optimized_allocation
        
        elif self.optimization_criterion == "Risk Parity EDaR":
            # Use skportfolio
            dailyIndividualReturns, dailyIndividualStd = self.basicMetrics()
            returns_data = dailyIndividualReturns.dropna()
            model = RiskBudgeting(
                    risk_measure=RiskMeasure.EDAR,
                    prior_estimator=EmpiricalPrior(
                        mu_estimator=ShrunkMu(covariance_estimator=GraphicalLassoCV(), 
                                              vol_weighted_target=True,
                                              method=ShrunkMuMethods.BODNAR_OKHRIN),
                        covariance_estimator=GraphicalLassoCV()),
                    portfolio_params=dict(name="Risk Parity EDaR"))
            
            model.fit_predict(returns_data)
            optimized_allocation = model.weights_  # pd.Series

            return optimized_allocation
        
    def portfolioReturn(self, weights):  
        return self.portfolioPerformance(weights)[0]

    def efficientOpt(self, returnTarget, constraintSet=(0, 1)):  
        numAssets = len(self.meanReturns)  
        constraints = (
            {
                "type": "eq",
                "fun": lambda x: self.portfolioReturn(x) - returnTarget,
            },  
            {"type": "eq", "fun": lambda x: np.sum(x) - 1},
        )  
        bounds = tuple(constraintSet for asset in range(numAssets))  
        effOpt = sc.minimize(
            self.portfolioVariance,
            numAssets * [1.0 / numAssets],
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        return effOpt

    def calculatedResults(self):
        if self.optimization_criterion in ["Hierarchical Risk Parity", 
                                           "Hierarchical Equal Risk Contribution",
                                           "Distributionally Robust CVaR",
                                           "Black Litterman Empirical",
                                           "Minimize EVaR",
                                           "Risk Parity EDaR"]:
            # For HRP and HERC
            optimized_allocation = self.optimization_function()

            # Compute the optimized returns and std
            weights = optimized_allocation
            optimized_returns, optimized_std = self.portfolioPerformance(weights)

            optimized_allocation = pd.DataFrame(
                weights,
                index=self.meanReturns.index,
                columns=["allocation"],
            )
        else:
            optimized_portfolio = (self.optimization_function())  
            optimized_returns, optimized_std = self.portfolioPerformance(
                optimized_portfolio["x"]
            )  
            optimized_allocation = pd.DataFrame(
                optimized_portfolio["x"],
                index=self.meanReturns.index,
                columns=["allocation"],
            ) 

        # Efficient Frontier
        std, ret, shar = self.simulations()
        efficientList = (
            []
        )  
        targetReturns = np.linspace(
            min(ret), max(ret), 100
        )  
        for target in targetReturns:
            efficientList.append(
                self.efficientOpt(target)["fun"]
            )  

        optimized_returns, optimized_std = round(optimized_returns * 100, 2), round(
            optimized_std * 100, 2
        )

        return (
            optimized_returns,
            optimized_std,
            optimized_allocation,
            efficientList,
            targetReturns,
        )

    def simulations(self):  
        noOfPortfolios = 10000
        numAssets = len(self.meanReturns)
        weight = np.zeros((noOfPortfolios, numAssets))
        expectedReturn = np.zeros(noOfPortfolios)
        expectedVolatility = np.zeros(noOfPortfolios)
        sharpeRatio = np.zeros(noOfPortfolios)  

        for k in range(noOfPortfolios):
            w = np.array(np.random.random(numAssets))  
            w = w / np.sum(w) 
            weight[k, :] = w  
            expectedReturn[k], expectedVolatility[k] = self.portfolioPerformance(weight[k].T)  
            sharpeRatio[k] = (expectedReturn[k] - self.riskFreeRate) / expectedVolatility[k]  

        return expectedVolatility, expectedReturn, sharpeRatio

    def EF_graph(self):

        fig, ax = plt.subplots(figsize=(10, 7))

        # Efficient Frontier
        ax.plot(
            [ef_std * 100 for ef_std in self.efficientList],
            [target * 100 for target in self.targetReturns],
            color="black",
            linestyle="-",
            linewidth=4,
            label="Efficient Frontier",
            zorder=1,
            alpha=0.9,
        )

        if self.optimization_criterion == "Maximize Sharpe Ratio":
            label_v = "Maximum Sharpe Ratio Portfolio"
        elif self.optimization_criterion == "Maximize Sortino Ratio":
            label_v = "Maximum Sortino Ratio Portfolio"
        elif self.optimization_criterion == "Minimize Volatility":
            label_v = "Minimum Volatility Portfolio"
        elif self.optimization_criterion == "Minimize Tracking Error":
            label_v = "Minimum Tracking Error Portfolio"
        elif self.optimization_criterion == "Maximize Information Ratio":
            label_v = "Maximum Information Ratio Portfolio"
        elif self.optimization_criterion == "Minimize Conditional Value-at-Risk":
            label_v = "Minimum CVaR Portfolio"
        elif self.optimization_criterion == "Hierarchical Risk Parity":
            label_v = "Hierarchical Risk Parity Portfolio"
        elif self.optimization_criterion == "Hierarchical Equal Risk Contribution":
            label_v = "Hierarchical Equal Risk Contribution Portfolio"
        elif self.optimization_criterion == "Distributionally Robust CVaR":
            label_v = "Distributionally Robust CVaR"
        elif self.optimization_criterion == "Black Litterman Empirical":
            label_v = "Black Litterman Empirical"
        elif self.optimization_criterion == "Minimize EVaR":
            label_v = "Minimize EVaR"
        elif self.optimization_criterion == "Risk Parity EDaR":
            label_v = "Risk Parity EDaR"
        ## Max Sharpe Ratio
        ax.scatter(
            [self.optimized_std],
            [self.optimized_returns],
            color="orange",
            marker="o",
            s=150,
            label=label_v,
            edgecolors="darkgray",
            zorder=4,
        )

        ## Plot Random Portfolios
        expectedVolatility, expectedReturn, sharpeRatio = self.simulations()
        scatter = ax.scatter(
            expectedVolatility * 100,
            expectedReturn * 100,
            c=sharpeRatio,
            cmap="Blues",
            marker="o",
            zorder=0,
            s=40,
        )
        plt.colorbar(scatter, ax=ax, label="Sharpe Ratio")
        ax.set_xlabel("Annualised Volatility (%)")
        ax.set_ylabel("Annualised Return (%)")

        ax.legend()
        st.pyplot(fig)

    def allocCharts(self):
        ## Max Sharpe Ratio allocation
        sharpeChart = px.self.optimized_allocation.allocation()
        fig = px.pie(
            sharpeChart, values="allocation", names=self.optimized_allocation.index
        )
        return fig.show()

    def frontierStats(self):
        ## Summary Stats
        returns, std = self.basicMetrics()
        tickers = [i for i in self.optimized_allocation.index]
        ExpectedReturn = [f"{round(i*252*100, 2)} %" for i in self.meanReturns]
        StandardDeviation = [f"{round(i*np.sqrt(252)*100, 2)} %" for i in std]
        sharpeRatio = []
        for i, ret in enumerate(self.meanReturns):
            sharpe = (ret * 252 - self.riskFreeRate) / (std[i] * np.sqrt(252))
            sharpeRatio.append(round(sharpe, 2))

        df = pd.DataFrame(
            {
                "Tickers": tickers,
                "Expected Return": ExpectedReturn,
                "Standard Deviation": StandardDeviation,
                "Sharpe Ratio": sharpeRatio,
            }
        )

        ## Correlation Matrix
        matrix = returns.corr().round(decimals=2)
        matrix[""] = matrix.index
        matrix = matrix[[""] + [col for col in matrix.columns if col != ""]]
        matrix.columns = [stock.replace(".NS", "") for stock in matrix.columns]
        matrix[""] = [stock.replace(".NS", "") for stock in matrix[""]]

        return df, matrix