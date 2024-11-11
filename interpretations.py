import streamlit as st
from risk import RiskMetrics

def metric_info():
    st.write("")
    st.markdown("##### 1. Mean Return (Monthly and Annualized)")
    st.markdown("**Definition**: Mean Return is the average return your portfolio generates over a specific period, providing insight into typical earnings on a monthly or annual basis.")
    st.markdown("**Optimal Range**: Positive values are preferred, with higher means indicating stronger performance. It's crucial to assess returns in relation to the risk undertaken.")

    st.markdown("##### 2. Standard Deviation (Monthly and Annualized)")
    st.markdown("**Definition**: Standard Deviation gauges the portfolio's volatility by measuring how much returns fluctuate around the average. Higher values denote greater volatility.")
    st.markdown("**Optimal Range**: Lower standard deviation is generally desirable, signaling reduced risk. However, portfolios with high returns might tolerate slightly higher volatility.")

    st.markdown("##### 3. Downside Standard Deviation")
    st.markdown("**Definition**: This metric focuses on the volatility of negative returns, offering a clearer picture of potential losses.")
    st.markdown("**Optimal Range**: Lower values are better, indicating fewer and less severe downside deviations.")

    st.markdown("##### 4. Maximum Drawdown")
    st.markdown("**Definition**: Maximum Drawdown measures the largest peak-to-trough decline in your portfolio's value, highlighting the worst loss from a historical peak before recovery.")
    st.markdown("**Optimal Range**: Lower drawdowns are preferable, reflecting smaller losses and better capital preservation.")

    st.markdown("##### 5. Beta")
    st.markdown("**Definition**: Beta assesses your portfolio's sensitivity to market movements. A beta above 1 means higher volatility than the market, while below 1 indicates lower volatility.")
    st.markdown("**Optimal Range**: A beta around 1 suggests volatility similar to the market. Lower betas reduce risk but may also lower potential returns, whereas higher betas increase both risk and potential returns.")

    st.markdown("##### 6. Alpha")
    st.markdown("**Definition**: Alpha measures the portfolio's excess return over its benchmark, adjusted for risk. Positive alpha signifies outperformance, while negative alpha indicates underperformance.")
    st.markdown("**Optimal Range**: Positive alpha is desirable, showing that the portfolio is outperforming the benchmark after accounting for risk.")

    st.markdown("##### 7. Sharpe Ratio")
    st.markdown("**Definition**: The Sharpe Ratio evaluates risk-adjusted returns by showing the excess return per unit of total risk.")
    st.markdown("**Optimal Range**: Higher Sharpe Ratios are better, indicating more efficient return generation relative to risk. Ratios above 1 are generally considered good.")

    st.markdown("##### 8. Sortino Ratio")
    st.markdown("**Definition**: Similar to the Sharpe Ratio, the Sortino Ratio focuses solely on downside risk, measuring returns relative to negative volatility.")
    st.markdown("**Optimal Range**: Higher Sortino Ratios are preferred, reflecting better returns per unit of downside risk. Ratios above 2 are excellent.")

    st.markdown("##### 9. Treynor Ratio")
    st.markdown("**Definition**: The Treynor Ratio assesses risk-adjusted returns relative to systematic risk (beta), indicating excess return per unit of market risk.")
    st.markdown("**Optimal Range**: Higher Treynor Ratios signify better performance per unit of market risk.")

    st.markdown("##### 10. Calmar Ratio")
    st.markdown("**Definition**: The Calmar Ratio compares the portfolio's annualized return to its maximum drawdown, providing insight into performance relative to drawdown risk.")
    st.markdown("**Optimal Range**: Higher Calmar Ratios are favorable, indicating better returns for the level of drawdown risk. Ratios above 1 are considered good.")

    st.markdown("##### 11. Tracking Error")
    st.markdown("**Definition**: Tracking Error measures the standard deviation of the portfolio's excess returns compared to its benchmark, indicating how closely it follows the benchmark.")
    st.markdown("**Optimal Range**: Lower Tracking Errors are ideal for portfolios aiming to mimic the benchmark closely. Higher values may be acceptable for actively managed portfolios seeking higher returns.")

    st.markdown("##### 12. Information Ratio")
    st.markdown("**Definition**: The Information Ratio evaluates the consistency of the portfolio's excess returns over the benchmark, adjusted for Tracking Error.")
    st.markdown("**Optimal Range**: Higher Information Ratios are better, showing consistent outperformance relative to the benchmark. Ratios above 0.5 are generally seen as good.")

    st.markdown("##### 13. Skewness")
    st.markdown("**Definition**: Skewness measures the asymmetry of the return distribution. Positive skew indicates more frequent small gains and occasional large losses, while negative skew suggests the opposite.")
    st.markdown("**Optimal Range**: Positive skewness is typically preferred, as it implies a higher likelihood of gains and fewer extreme losses.")

    st.markdown("##### 14. Excess Kurtosis")
    st.markdown("**Definition**: Excess Kurtosis assesses the \"tailedness\" of the return distribution. Higher values indicate more frequent extreme returns, while lower values suggest fewer extremes.")
    st.markdown("**Optimal Range**: Lower Excess Kurtosis is generally favored, indicating a more stable return distribution with fewer extreme events.")

    st.markdown("##### 15. Positive Periods")
    st.markdown("**Definition**: Positive Periods count the number of periods where the portfolio achieved positive returns, reflecting performance consistency.")
    st.markdown("**Optimal Range**: A higher percentage of positive periods is desirable, showing consistent positive performance. However, this should be evaluated alongside other risk and return metrics.")

def var_info():
    st.markdown("##### Value at Risk (VaR)")
    st.markdown("**Definition**: VaR estimates the maximum potential loss of the portfolio over a specified period at a given confidence level. For instance, a 95% VaR indicates the maximum expected loss with a 5% probability.")
    st.markdown("**Optimal Range**: Lower VaR values signify reduced risk, as they suggest smaller potential losses. The acceptable VaR level depends on the investor's risk tolerance, but minimizing VaR without significantly sacrificing returns is typically the goal.")

    st.markdown("##### Conditional Value at Risk (CVaR)")
    st.markdown("**Definition**: CVaR calculates the average loss exceeding the VaR threshold, providing insight into the expected losses in extreme adverse scenarios.")
    st.markdown("**Optimal Range**: Lower CVaR values are preferable, indicating smaller average losses in extreme cases. Minimizing CVaR helps manage tail risk and protect against severe downturns.")

def appinfo():
    st.markdown("This application optimizes your stock portfolio using various strategies to identify the most efficient allocation based on selected criteria. Simply input your stock tickers, choose a historical date range, and the app will analyze performance and provide tailored allocation recommendations to help you achieve your investment objectives.")

def optimization_strategies_info():
    st.write("")
    st.markdown("**Sharpe Ratio**: Evaluates return per unit of total risk, ideal for investors aiming to maximize returns relative to risk. A higher Sharpe Ratio reflects more effective risk-adjusted return generation.")
    st.markdown("**Volatility**: Measured by the standard deviation of portfolio returns, this metric is key for investors focused on minimizing fluctuations. Lower volatility signifies a more stable investment, aligning with conservative strategies.")
    st.markdown("**Sortino Ratio**: Focuses on downside risk by measuring returns per unit of negative volatility. It's beneficial for investors prioritizing capital preservation and minimizing significant losses. A higher Sortino Ratio indicates better performance during adverse conditions.")
    st.markdown("**Tracking Error**: Assesses how closely a portfolio's returns follow its benchmark. Suitable for investors seeking to align their portfolio with a benchmark index. Lower Tracking Error signifies greater consistency with benchmark performance.")
    st.markdown("**Information Ratio**: Measures excess returns over a benchmark relative to the active risk taken. Ideal for active investors striving to outperform a benchmark while managing risk. A higher Information Ratio indicates successful active management.")
    st.markdown("**Conditional Value-at-Risk (CVaR)**: Estimates potential losses in extreme market conditions, focusing on worst-case scenarios. Essential for risk-averse investors aiming to protect capital against severe downturns. Lower CVaR values indicate better risk mitigation.")
    st.markdown("**Hierarchical Risk Parity (HRP)**: An advanced portfolio optimization technique that builds a diversified portfolio by hierarchically clustering assets based on their correlations. It allocates weights to balance risk across clusters, reducing the reliance on traditional covariance matrices. Ideal for investors seeking diversification and stability without the assumptions of mean-variance optimization.")
    st.markdown("**Hierarchical Equal Risk Contribution (HERC)**: Enhances the HRP method by ensuring that each cluster contributes equally to the overall portfolio risk. It combines hierarchical clustering with equal risk contribution principles, offering a balanced approach to diversification. Suitable for investors aiming for an equal distribution of risk among asset clusters.")
    st.markdown("**Distributionally Robust CVaR**: A conservative risk management approach that accounts for uncertainty in the distribution of asset returns when calculating CVaR. It provides more robust risk estimates under distributional ambiguity, ideal for investors who want to protect against model misspecification and extreme market movements.")
    st.markdown("**Black-Litterman Model with Empirical Prior**: The Black-Litterman model is an advanced portfolio optimization technique that blends market equilibrium returns with an investor's unique views to generate a new set of expected returns. Using empirical distribution as the prior means that historical market data serves as the foundation for baseline expectations. Investors adjust these expectations based on their own insights or forecasts. This method mitigates the impact of estimation errors in expected returns and reduces portfolio sensitivity to input assumptions. Ideal for investors who wish to incorporate their views into the optimization process while maintaining a balance with market consensus. Benefits include more stable and diversified portfolios that reflect both historical data and the investor's perspectives.")
    st.markdown("**Minimize EVaR (Entropic Value at Risk)**: Focuses on reducing potential extreme losses by considering the worst-case scenarios in a more conservative way than traditional risk measures. EVaR provides a tighter estimate of potential losses during extreme market conditions. Ideal for highly risk-averse investors who want to safeguard their portfolios against rare but severe downturns. By minimizing EVaR, you aim to build a portfolio that's better protected against extreme risks, offering greater peace of mind in uncertain markets.")
    st.markdown("**Risk Budgeting with EDAR (Entropic Drawdown at Risk)**: Allocates risk among assets by minimizing the sum of individual entropic drawdowns, focusing on managing potential severe declines in asset values. EDAR measures the worst-case drawdown in a conservative manner, capturing extreme negative movements. Ideal for investors who want to balance risk contributions across portfolio components while safeguarding against significant losses. This strategy ensures that each asset's risk is budgeted according to predefined levels, promoting a diversified and risk-aware portfolio.")
    st.markdown("*Benchmark: S&P 500*")

def model_parameters_info():
    st.markdown("### Understanding Model Parameters")
    st.write("")
    st.markdown("**Hidden Dimension**: Think of this as the 'depth' of understanding that the model can develop about historical market patterns. A larger hidden dimension allows the model to pick up on more subtle patterns and nuances, but also makes the model more complex. For simpler patterns, a smaller hidden dimension can be sufficient.")
    st.markdown("**Number of Flows**: Imagine this as a series of 'lenses' through which the model views the data, one layer at a time. Each flow layer transforms the data slightly, helping the model build a more accurate and flexible understanding of market behavior. More flows mean a deeper and more flexible model but can take longer to train.")

def forecast_metrics_info():
    st.markdown("### Understanding Model Evaluation Metrics")
    st.write("")
    st.markdown("**Mean Absolute Error (MAE)**: MAE measures how “off” our predictions are from the actual values by calculating the average difference between each prediction and its actual value.")
    st.markdown("**Root Mean Squared Error (RMSE)**: RMSE, like MAE, measures prediction error, but it gives more weight to larger errors by squaring them first. This makes it especially useful for scenarios where larger errors are more problematic.")
    st.markdown("**Regression Coverage Score (RCS)**: RCS tells us how often our prediction intervals contain the actual values. If our model's interval covers the true outcome frequently, RCS will be high.")
    st.markdown("**Regression Mean Width Score (RMWS)**: RMWS measures the average width of our prediction intervals, which shows the distance between the high and low bounds of our prediction range.")
    st.markdown("**Coverage Width-Based Criterion (CWC)**: CWC evaluates prediction intervals by balancing both their coverage and width, rewarding narrow intervals that include true outcomes and penalizing those that don't.")
