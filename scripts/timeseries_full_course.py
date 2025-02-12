# %%
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from warnings import filterwarnings

filterwarnings('ignore')

apple_stock_data = yf.download(tickers='AAPL', start='2024-01-01')
apple_stock_data

# %%
apple_stock_data.index

# %%
plt.figure(figsize=(12,8))
plt.plot(apple_stock_data.index, apple_stock_data['Close'], label='Close price', color = '#ff9140')
plt.title('AAPL stock')
plt.xlabel('Dates')
plt.ylabel('Stock price')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# # Classical timeseries decomposition

# %%
additive_decomposition = seasonal_decompose(x=apple_stock_data['Close'], model='additive', period=30)
additive_decomposition

# %%
trend_additive = additive_decomposition.trend
seasonal_additive = additive_decomposition.seasonal
residual_additive = additive_decomposition.resid

trend_additive, seasonal_additive, residual_additive

# %%
plt.figure(figsize=(13,17))
plt.subplot(411)
plt.plot(apple_stock_data['Close'], label = 'Original', color = '#ff9140')
plt.legend()

plt.subplot(412)
plt.plot(trend_additive, label = 'Trend', color = '#ff9140')
plt.legend()

plt.subplot(413)
plt.plot(seasonal_additive, label = 'Seasonal', color = '#ff9140')
plt.legend()

plt.subplot(414)
plt.plot(residual_additive, label = 'Residual', color = '#ff9140')
plt.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# # STL decomposition
# 
# STL (Seasonal-Trend decomposition using Loess) is a versatile and powerful technique for breaking down a time series into three main components:
# 
# 1. **Seasonal Component**: Repeating patterns or cycles in the data, like daily, monthly, or yearly cycles.
# 2. **Trend Component**: Long-term movement in the data, showing whether the values are increasing, decreasing, or stable over time.
# 3. **Residual Component**: The "noise" or "error" component, capturing any variation that is not explained by the seasonal or trend components.
# 
# ### Key Features of STL Decomposition:
# - **Seasonal-Trend Loess (STL)** uses *Loess* (locally estimated scatterplot smoothing) to smooth out data. 
# - **Flexible**: STL allows for custom seasonality windows, making it suitable for time series data with non-fixed seasonality.
# - **Robust to Noise**: Can handle noise in data, providing a clearer view of trends and seasonality.
# 
# ### How STL Decomposition Works
# STL decomposition applies Loess smoothing to separate out the trend and seasonal components iteratively, and then extracts the residuals.
# 
# ### Example: Applying STL Decomposition in Python
# To illustrate STL decomposition, let's use a sample time series. Here’s how to do it with Python’s `statsmodels` library.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

# Generate a sample time series with a trend, seasonality, and noise
np.random.seed(42)
time = pd.date_range(start='2020-01-01', periods=365, freq='D')
trend = np.linspace(10, 20, 365)  # Linear trend
seasonal = 10 * np.sin(2 * np.pi * time.dayofyear / 365)  # Yearly seasonality
noise = np.random.normal(scale=2, size=365)  # Random noise
data = trend + seasonal + noise

# Create a DataFrame
df = pd.DataFrame({'Date': time, 'Value': data}).set_index('Date')

# Apply STL decomposition
stl = STL(df['Value'], seasonal=13)
result = stl.fit()

# Plot the components
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
ax1.plot(df['Value'], label='Original Data')
ax1.legend(loc='upper left')
ax2.plot(result.trend, label='Trend', color='orange')
ax2.legend(loc='upper left')
ax3.plot(result.seasonal, label='Seasonal', color='green')
ax3.legend(loc='upper left')
ax4.plot(result.resid, label='Residual', color='red')
ax4.legend(loc='upper left')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Explanation of the Plot
# 1. **Original Data**: Shows the entire time series with trend, seasonality, and noise combined.
# 2. **Trend Component**: Extracts the long-term upward movement.
# 3. **Seasonal Component**: Shows the repeating pattern over time.
# 4. **Residual Component**: Shows the remaining noise after removing the trend and seasonality.
# 
# ### Summary
# STL decomposition is useful because it allows for:
# - **Analyzing** the behavior of the trend and seasonality independently.
# - **Cleaning** the data by removing seasonality or detrending.
# - **Modeling** each component separately or combining them into predictive models.

# %% [markdown]
# # Classical and STL usage
# Choosing between **Classical Decomposition** and **STL Decomposition** depends on the characteristics of the time series and the specific goals of the analysis.
# 
# ### 1. Classical Decomposition
# Classical decomposition splits a time series into **additive** or **multiplicative** components (Trend, Seasonal, and Residual). It works well under certain conditions but is limited in flexibility.
# 
# #### When to Use Classical Decomposition:
# - **Fixed, Regular Seasonality**: The time series has a stable and known seasonal pattern (e.g., monthly, quarterly) that does not change over time.
# - **Simple Patterns**: The time series has clear and consistent seasonality and trend, without significant variation or noise.
# - **Quick Analysis or Exploration**: If you need a fast, straightforward decomposition to explore general trends and seasonal components without high accuracy demands.
# - **Stationary Seasonality**: If seasonality doesn’t vary in magnitude or frequency over time (e.g., temperature data with a clear annual cycle).
# 
# > **Example**: Monthly retail sales data with predictable seasonal peaks (e.g., December holiday shopping).
# 
# #### Limitations of Classical Decomposition:
# - **Inflexible Seasonality**: Classical decomposition cannot handle time series where seasonal patterns shift over time.
# - **Non-Adaptability**: It does not handle complex, non-linear trends well and struggles with irregular or non-fixed seasonality.
# 
# ### 2. STL Decomposition
# STL (Seasonal-Trend decomposition using Loess) is more versatile and flexible, as it doesn’t require the seasonality to be fixed. It handles complex time series better and can deal with irregularities in seasonal patterns.
# 
# #### When to Use STL Decomposition:
# - **Varying Seasonality**: The seasonality pattern changes over time in magnitude or frequency (e.g., retail sales influenced by changing consumer behavior).
# - **Noise in Data**: STL is robust to noise, making it suitable for noisy time series.
# - **Non-linear Trends**: STL handles non-linear and non-stationary trends better than classical methods.
# - **Customization Needs**: If you need to adjust the seasonality or trend smoothness, STL allows fine-tuning of these parameters.
# - **Missing Values**: STL is more tolerant of missing data than classical decomposition, making it more practical for real-world applications.
# 
# > **Example**: Hourly website traffic data with changing patterns over months and years or monthly energy consumption with varying seasonal effects.
# 
# #### Limitations of STL:
# - **Computationally Intensive**: STL requires more computational resources, especially for large datasets, as it iteratively smooths components.
# - **Complexity**: The results can be harder to interpret due to the number of tuning parameters.
# 
# ### Summary
# | **Criterion**                | **Classical Decomposition**                | **STL Decomposition**                     |
# |------------------------------|--------------------------------------------|-------------------------------------------|
# | **Seasonality**              | Fixed and regular                          | Can vary over time                        |
# | **Trend**                    | Simple, stationary                         | Complex, non-linear                       |
# | **Noise Tolerance**          | Less tolerant                              | More robust to noise                      |
# | **Data Requirements**        | Complete data                              | Tolerates missing values                  |
# | **Computational Resources**  | Low                                        | Higher (more computationally intensive)   |
# | **Customization**            | Limited                                    | Highly customizable                       |
# 
# In summary, **Classical Decomposition** is ideal for quick, exploratory analysis of stable time series with fixed seasonality, while **STL Decomposition** excels for complex, noisy time series with flexible seasonal patterns. If the seasonality or trend is expected to change over time or if the series has irregular patterns, STL is generally a better choice.

# %% [markdown]
# # Stationarity tests

# %% [markdown]
# There're following stationarity statistical tests:
# <ol type = "a">
#    <li>ADF test</li>
#    <li>KPSS test</li>
#    <li>KS test/K-S test</li>
# </ol>
# 
# 
# Let’s go through each of the stationarity tests one by one to understand what they do and how they work.
# 
# ---
# 
# ### a. **ADF Test (Augmented Dickey-Fuller Test)**
# 
# The **Augmented Dickey-Fuller (ADF) Test** is one of the most popular tests for checking if a time series is **stationary**.
# 
# #### What it Does:
# The ADF test looks for a **unit root** in the data, which is a characteristic of non-stationary series. If the data has a unit root, it means the time series has a trend, making it non-stationary. 
# 
# #### How it Works:
# 1. **Hypotheses**:
#    - **Null Hypothesis (H₀)**: The series has a unit root (it’s non-stationary).
#    - **Alternative Hypothesis (H₁)**: The series does not have a unit root (it’s stationary).
#    
# 2. **Test Statistic**:
#    - The test gives a test statistic value that is compared to critical values. If the test statistic is less than the critical value (or if the p-value is below a chosen significance level, usually 0.05), you reject the null hypothesis, suggesting that the series is stationary.
# 
# 3. **Interpretation**:
#    - If **p-value < 0.05**: Reject the null hypothesis; the series is likely stationary.
#    - If **p-value > 0.05**: Fail to reject the null hypothesis; the series is likely non-stationary.
# 
# #### Example Use:
# ```python
# from statsmodels.tsa.stattools import adfuller
# 
# # Apply ADF test
# result = adfuller(time_series)
# print("ADF Statistic:", result[0])
# print("p-value:", result[1])
# 
# if result[1] < 0.05:
#     print("Series is stationary")
# else:
#     print("Series is non-stationary")
# ```
# 
# ---
# 
# ### b. **KPSS Test (Kwiatkowski-Phillips-Schmidt-Shin Test)**
# 
# The **KPSS Test** is another test for stationarity, but it works differently from the ADF test. Instead of looking for a unit root, it tests if the series is **trend-stationary**.
# 
# #### What it Does:
# The KPSS test checks if a time series has a **stationary trend** or not. It’s often used to confirm results from the ADF test because the two tests work in opposite ways.
# 
# #### How it Works:
# 1. **Hypotheses**:
#    - **Null Hypothesis (H₀)**: The series is stationary (no unit root).
#    - **Alternative Hypothesis (H₁)**: The series is non-stationary (has a unit root).
#    
# 2. **Test Statistic**:
#    - Similar to the ADF test, KPSS produces a test statistic and a critical value. However, here, if the test statistic is greater than the critical value (or if the p-value is below 0.05), you reject the null hypothesis and conclude that the series is non-stationary.
# 
# 3. **Interpretation**:
#    - If **p-value < 0.05**: Reject the null hypothesis; the series is likely non-stationary.
#    - If **p-value > 0.05**: Fail to reject the null hypothesis; the series is likely stationary.
# 
# #### Example Use:
# ```python
# from statsmodels.tsa.stattools import kpss
# 
# # Apply KPSS test
# result = kpss(time_series)
# print("KPSS Statistic:", result[0])
# print("p-value:", result[1])
# 
# if result[1] < 0.05:
#     print("Series is non-stationary")
# else:
#     print("Series is stationary")
# ```
# 
# ---
# 
# ### c. **KS Test (Kolmogorov-Smirnov Test)**
# 
# The **Kolmogorov-Smirnov (KS) Test** is a general-purpose test that’s used to check if a sample follows a specific distribution. While it’s not a stationarity test per se, it can sometimes be used in time series analysis to check if the distribution of values has changed over time (e.g., in different periods).
# 
# #### What it Does:
# The KS test compares the distribution of your time series data to a specific reference distribution (e.g., normal distribution) to see if they’re the same. 
# 
# #### How it Works:
# 1. **Hypotheses**:
#    - **Null Hypothesis (H₀)**: The sample data follows the reference distribution.
#    - **Alternative Hypothesis (H₁)**: The sample data does not follow the reference distribution.
#    
# 2. **Test Statistic**:
#    - The KS test calculates the maximum distance between the cumulative distribution function (CDF) of the sample and the CDF of the reference distribution.
#    - If the test statistic is large (or p-value is low), it suggests that the data does not follow the reference distribution.
# 
# 3. **Interpretation**:
#    - If **p-value < 0.05**: Reject the null hypothesis; the data does not follow the reference distribution.
#    - If **p-value > 0.05**: Fail to reject the null hypothesis; the data follows the reference distribution.
# 
# #### Example Use:
# ```python
# from scipy.stats import kstest
# 
# # Apply KS test
# result = kstest(time_series, 'norm')  # Compare to normal distribution
# print("KS Statistic:", result.statistic)
# print("p-value:", result.pvalue)
# 
# if result.pvalue < 0.05:
#     print("Data does not follow the normal distribution")
# else:
#     print("Data follows the normal distribution")
# ```
# 
# ---
# 
# ### Summary
# 
# - **ADF Test**: Looks for unit roots to identify if a series is non-stationary.
# - **KPSS Test**: Tests for trend-stationarity by looking for the presence of a trend.
# - **KS Test**: Compares the data’s distribution to a reference distribution (e.g., normal distribution) and isn’t specifically a stationarity test but can be useful for analyzing distributional changes.
# 
# Each test gives insights into different aspects of time series stationarity and distribution, helping to decide the right pre-processing steps before modeling.

# %% [markdown]
# ## Other examples of ADF, KPSS, K-S test

# %% [markdown]
# ### ADF test example

# %%
from statsmodels.tsa.stattools import adfuller

adf_test_val = adfuller(apple_stock_data['Close'])

print("ADF statistic: ", adf_test_val[0])
print("P-value: ", adf_test_val[1])
print("Critical values: ")
for key, val in adf_test_val[4].items():
    print(f"  {key}: {val}")

# %% [markdown]
# ### KPSS test example

# %%
from statsmodels.tsa.stattools import kpss

kpss_test = kpss(x=apple_stock_data['Close'],regression='ct')

print("KPSS statistics: ", kpss_test[0])
print("KPSS p-value: ", kpss_test[1])
print("Critical value of kpss test: ")
for key, val in kpss_test[3].items():
    print(f"\t{key}: {val}")

# %% [markdown]
# ### KS Test

# %% [markdown]
# #### By me
# This method is used with the referenced distribution as input as well.

# %%
from statsmodels.stats.diagnostic import kstest_fit

kstest_val = kstest_fit(x = apple_stock_data['Close'], dist='exp')

print("KS test statistic: ", kstest_val[0])
print("KS test p-value: ", kstest_val[1])

# %% [markdown]
# #### Using scipy stats

# %% [markdown]
# ##### KS1 sample test

# %%
from scipy.stats import ks_1samp, norm

ks1_val = ks_1samp(x=apple_stock_data['Close'], cdf=norm.cdf)
print("KS1 statistic:", ks1_val.statistic[0])
print("P-value:", ks1_val.pvalue[0])

# %%
from scipy.stats import ks_1samp, expon

ks1_val = ks_1samp(x=apple_stock_data['Close'], cdf=expon.cdf)
print("KS1 statistic:", ks1_val.statistic[0])
print("P-value:", ks1_val.pvalue[0])

# %% [markdown]
# ##### KS test with generated strict and non-strict stationary series

# %%
import numpy as np
from scipy.stats import ks_2samp
from numpy.typing import ArrayLike

np.random.seed(42)
n = 500

strict_stationary_series = np.random.normal(0, 1, n)
non_strict_stationary_series = np.concatenate([
    np.random.normal(0, 2, n // 2),
    np.random.normal(0, 2, n // 2)
])

def ks2_samp(series: ArrayLike):
    """
    Test of stationarity using ks testing technique
    for stationarity
    """

    split = len(series) // 2
    series_first_half = series[:split]
    series_second_half = series[split:]
    kstats, pval = ks_2samp(series_first_half, series_second_half)
    return kstats, pval

strict_ks2_tats, strict_ks2_pval = ks2_samp(strict_stationary_series)
print("Strict stationary ks2 stats:", strict_ks2_tats)
print("Strict stationarity ks2 p value:", strict_ks2_pval)

weak_ks2_stats, weak_pval = ks2_samp(non_strict_stationary_series)
print("Strict stationary ks2 stats:", weak_ks2_stats)
print("Strict stationarity ks2 p value:", weak_pval)

# %% [markdown]
# # Converting non-stationary data to stationary
# 
# ## Differencing
# Subtracts the current occurrence of a series' output from its previous output. Can be done up-to nth order, where n could be the order of seasonality date.
# 
# ## Transforming
# Stabilizes the variance of a timeseries.
# * Log transformation
# * Exponential transformation (sqrt, cubic-root etc.)
# * Box-cox (Combination of log and exponential) transformation
# 
# ## De-trending
# ### a. Linear de-trending
# * Fit a regression line over the trend line.
# * Subtract the non-fitted points from the trend line data points, you're done.
# 
# ### b. Moving average de-trending
# * Select a window size, to select the number of elements to make sum of.
# * Add the sum of the number, adjacent to the first number; coming in the scope of the window.
# 
# ## Seasonal adjustment
# Removing seasonal component from the series, using `STL`.
# 
# ---
# 
# Below is the code demo of each of them

# %% [markdown]
# ## Stationarity checks

# %%
from statsmodels.tsa.stattools import adfuller
from numpy.typing import ArrayLike

def adf_stationarity_test(series: ArrayLike):
    adf_test_val = adfuller(series)

    print("ADF statistic: ", adf_test_val[0])
    print("P-value: ", adf_test_val[1])
    print("Critical values: ")
    for key, val in adf_test_val[4].items():
        print(f"  {key}: {val}")

# %%
from scipy.stats import ks_2samp

def ks2_samp(series: ArrayLike):
    """
    Test of stationarity using ks testing technique
    for stationarity
    """

    split = len(series) // 2
    series_first_half = series[:split]
    series_second_half = series[split:]
    kstats, pval = ks_2samp(series_first_half, series_second_half)
    print("Kstats:", kstats[0])
    print("P-value of ks test:", pval[0])

# %%
# Actual demonstration of the stationality check
# Differencing
prices = apple_stock_data['Close']
differenced_price = prices.diff()
differenced_price

# %%
prices

# %%
adf_stationarity_test(differenced_price.dropna())

# %%
ks2_samp(differenced_price.dropna())

# %%
import numpy as np
# log transform
logged_transform = np.log(prices)
logged_transform

# %%
adf_stationarity_test(logged_transform)

# %%
sqrt_transform = np.sqrt(prices)
sqrt_transform

# %%
adf_stationarity_test(sqrt_transform)

# %%
from scipy.stats import boxcox

boxcox_transform = boxcox(prices['AAPL'].values)
boxcox_transform

# %%
adf_stationarity_test(boxcox_transform[0])

# %%
import numpy as np

# Convert the 'AAPL' column in 'prices' to a Series
prices_series = prices['AAPL']

# Fit a linear trend line
trendline_coeff = np.polyfit(np.arange(len(prices_series)), prices_series, 1)
trendline = np.polyval(trendline_coeff, np.arange(len(prices_series)))

# Calculate the detrended prices
detrended_prices = prices_series - trendline

# %%
adf_stationarity_test(detrended_prices)

# %%
plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(prices_series)), prices_series, label='Actual Prices', color='orange')
plt.plot(np.arange(len(prices_series)), trendline, label='Trendline', color='red', linestyle='--')
plt.xlabel('Price series')
plt.ylabel('Price')
plt.title('Actual Prices and Trendline')
plt.legend()
plt.show()

# %%
window_size = 12
prices_ma = prices.rolling(window = window_size).mean()
# prices_ma.dropna(inplace=True)
prices_ma_detrended = prices - prices_ma
# prices_ma_detrended.dropna(inplace=True)
prices_ma_detrended

# %%
adf_stationarity_test(prices_ma_detrended.dropna())

# %%
plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(prices_series)), prices_series, label='Actual Prices', color='orange')
plt.plot(np.arange(len(prices_series)), prices_ma, label='Moving Average', color='red', linestyle='--')
plt.xlabel('Price series')
plt.ylabel('Price')
plt.title('Actual Prices and Moving Average')
plt.legend()
plt.show()

# %%
from statsmodels.tsa.seasonal import seasonal_decompose

prices_decomposition = seasonal_decompose(x=prices, model='additive', period=28)
deseasoned_prices = prices / prices_decomposition.seasonal
deseasoned_prices.dropna()

# %%
from statsmodels.tsa.seasonal import STL

prices_stl = STL(prices['AAPL'], seasonal=13, period=30)
prices_decomposed = prices_stl.fit()

prices_de_seasoned = prices / prices_decomposed.seasonal
prices_de_seasoned.dropna()

# %%
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np

# Ensure `prices` is a single column Series
prices_series = prices['AAPL']

# Check for and handle missing values
prices_series = prices_series.fillna(method='ffill').fillna(method='bfill')

# Perform decomposition with a suitable period
try:
    decomposition = seasonal_decompose(prices_series, model='additive', period=5)  # Adjust period as needed
    deseasoned_prices = prices_series - decomposition.seasonal
    deseasoned_prices = deseasoned_prices.dropna()  # Remove NaNs, if any, after decomposition
except ValueError as e:
    print("Error during decomposition:", e)

# Check if deseasoned data is now stationary with ADF
from statsmodels.tsa.stattools import adfuller
result = adfuller(deseasoned_prices)
print("ADF test statistic:", result[0])
print("p-value:", result[1])

# %%
first_diff = prices_series.diff().dropna()
second_diff = first_diff.diff().dropna()
adf_result = adfuller(second_diff)
print("ADF test p-value after second differencing:", adf_result[1])

# %%
3.3172675871544726e-11 < 0.05

# %%
second_diff.to_frame().plot()

# %%
# MinMaxScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler

min_max_scaled = MinMaxScaler().fit_transform(prices)
std_scaled = StandardScaler().fit_transform(prices)

# %%
adf_stationarity_test(min_max_scaled)

# %%
adf_stationarity_test(std_scaled)

# %% [markdown]
# # White noise and random walk
#
# ***What is white noise?***
#
# A series with no seasonality, trend. Statistically speaking, it has:
#   - Constant mean
#   - Constant variance
#   - No autocorrelation
#
# ***What is random walk?***
#
# A random walk in a series is the sum of the previous value(s) to reach the latest or current value, it has a cumulative frequency pattern.
#
# It's generally seen in stock prices. It has following properties:
#  - It's predictable, since `y_t_current = y_t_current-1 + error_t_current`. Here `error_t_current` is white noise.
#  - It has first difference as a stationary series.
#  - It has a varying mean & variance over time.
#  - After making the random walk series a stationary one, it is been prepared to be used for training the forecasting model.

# %% [markdown]
# ## Simulating and testing white noise and random walk

# %%
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch.unitroot import VarianceRatio
import numpy as np

np.random.seed(0)
n = 1000

white_noise = np.random.normal(0,1,n)

random_noise = np.random.normal(0,1,n)
random_walk = np.cumsum(random_noise)

"""
    a. The Variance Ratio Test is specifically designed to detect random walk behavior.
    b. It compares the variance of returns over multiple periods to the variance of returns over single periods.
    c. For a random walk, variances scale linearly with time, so the variance ratio should equal 1. Significant deviations
    from 1 suggest non-random walk behavior.
"""
lj_df = acorr_ljungbox(white_noise, lags=[10])
vr_ratio = VarianceRatio(random_walk)

print("Ljung box test:", lj_df)
print(f"Variance ratio statistics:{vr_ratio.stat}")
print(f"Variance ratio p-value:{vr_ratio.pvalue}")

# %% [markdown]
# Based on the results, let’s break down the interpretation of each test:
# 
# ### 1. **Ljung-Box Test (for White Noise in `white_noise`)**
#    - **Ljung-Box Test Output**: 
#      - **Statistic (`lb_stat`)**: 14.025574
#      - **P-value (`lb_pvalue`)**: 0.171828
#    - **Interpretation**:
#      - The p-value is greater than 0.05 (significance level), so we fail to reject the null hypothesis that the series has no autocorrelation.
#      - This suggests that `white_noise` is consistent with white noise, as it does not show significant autocorrelation.
# 
# ### 2. **Variance Ratio Test (for Random Walk in `random_walk`)**
#    - **Variance Ratio Test Output**:
#      - **Statistic**: -1.1593266582917727
#      - **P-value**: 0.2463230593870811
#    - **Interpretation**:
#      - The p-value here is also greater than 0.05, indicating that we fail to reject the null hypothesis that `random_walk` is indeed a random walk.
#      - The negative statistic is typical in variance ratio tests, especially when the series is very close to a random walk, as we expect a variance ratio around 1.
# 
# ### Overall Summary
# - **White Noise Series (`white_noise`)**: The Ljung-Box test results suggest `white_noise` is uncorrelated (consistent with white noise).
# - **Random Walk Series (`random_walk`)**: The Variance Ratio test results suggest `random_walk` behaves like a random walk. 
# 
# The test's outputs support the series' intended behaviors: the `white_noise` appears uncorrelated (consistent with white noise), and `random_walk` behaves in a way consistent with a random walk.

# %% [markdown]
# ## White noise checking for `second_diff`

# %% [markdown]
# ### ACF, PACF plots

# %%
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(second_diff)
plot_pacf(second_diff)
plt.show()

# %% [markdown]
# The partial autocorrelation plot for `second_diff` shows that it has high correlation of 1 with lag 0, and the rest of the plots having negative correlation, the 25th lag has correlation just above 0.00, while, the ACF plot show some correlation with correlation of 1 with lag 0 and other lags have some correlations, with some near 0.00; and lag 1,3,5,10,12,13,20,21,23 have low correlation than 0.00, in negative.

# %% [markdown]
# ### L-jung-box test

# %%
from statsmodels.stats.diagnostic import acorr_ljungbox

lb_test = acorr_ljungbox(second_diff, lags=[25], return_df=True)
lb_test

# %% [markdown]
# The Ljung-Box test results for `second_diff` show:
# 
# - **Test Statistic (`lb_stat`)**: 82.57
# - **P-value (`lb_pvalue`)**: 4.478297e-08, which is very close to zero.
# 
# ### Interpretation:
# 
# 1. **Low P-value**: The extremely low p-value (< 0.05) suggests that the null hypothesis of white noise is rejected. This implies that there is some significant autocorrelation remaining in the differenced series.
#   
# 2. **Implication**: Since the data still shows some autocorrelation, it’s not purely random and may retain some structure or patterns. In practice, this often suggests that there could be further adjustments to make if you’re aiming for a white noise result, such as additional differencing or considering an ARIMA model to model the remaining structure.
# 
# In summary, the second differenced data is not behaving as white noise due to this statistically significant result.

# %% [markdown]
# ### Decomposition of `second_diff`

# %%
second_diff_decomposed = seasonal_decompose(x=second_diff, period=30)
plt.figure(figsize=(10,7))
second_diff_decomposed.plot()
plt.show()

# %% [markdown]
# # Single-variate models

# %% [markdown]
# ## Simulating workable data

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set a time range from 2020 to 2023 with monthly frequency
date_range = pd.date_range(start='2020-01-01', end='2023-12-31', freq='M')

# Generate a seasonal component with a yearly period
seasonal_period = 12  # 12 months for yearly seasonality
seasonal_component = 10 * np.sin(2 * np.pi * np.arange(len(date_range)) / seasonal_period)

# Generate stationary random noise
np.random.seed(42)  # for reproducibility
stationary_noise = np.random.normal(loc=0, scale=2, size=len(date_range))

# Combine seasonality and noise to form the initial time series
combined_series = seasonal_component + stationary_noise

# Convert to a DataFrame
seasonal_stationary_df = pd.DataFrame({
    'Date': date_range,
    'Value': combined_series
}).set_index('Date')

# Differencing to make the series stationary if needed
seasonal_stationary_df['Differenced_Value'] = seasonal_stationary_df['Value'].diff().dropna()

# Plot the series to visualize
plt.figure(figsize=(12, 6))
plt.plot(seasonal_stationary_df['Value'], label='Original Series with Seasonality')
plt.plot(seasonal_stationary_df['Differenced_Value'], label='Differenced Series (Stationary)')
plt.legend()
plt.title('Seasonal Stationary Series')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

seasonal_stationary_df

# %%
stationary_df_decompose = seasonal_decompose(x=seasonal_stationary_df['Value'])
plt.figure(figsize=(10,6))
stationary_df_decompose.plot()
plt.show()

# %%
adf_stationarity_test(seasonal_stationary_df['Value'])

# %%
seasonal_1st_diff = seasonal_stationary_df['Value'].diff().dropna()
adf_stationarity_test(seasonal_1st_diff)

# %%
seasonal_stat_2nd_diff = seasonal_1st_diff.diff().dropna()
adf_stationarity_test(seasonal_stat_2nd_diff)

# %%
# First, apply first-order differencing to remove the trend
trend_diff = seasonal_stationary_df['Value'].diff().dropna()

# Apply seasonal differencing (assuming yearly seasonality, period=12)
seasonal_diff = trend_diff.diff(periods=12).dropna()

# Check for stationarity with ADF test
from statsmodels.tsa.stattools import adfuller
adf_test = adfuller(seasonal_diff)
print(f"ADF Statistic: {adf_test[0]}")
print(f"p-value: {adf_test[1]}")

# Plot the final transformed series
plt.figure(figsize=(10, 6))
plt.plot(seasonal_diff, label='Deseasoned and Detrended Series')
plt.legend()
plt.title('Final Stationary Series')
plt.show()

# %% [markdown]
# ### Creating my own `stationary_ts_df`

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess

# Parameters for a stationary AR(1) process
np.random.seed(42)
ar = np.array([1, -0.7])  # AR coefficient (ensures stationarity with |0.7| < 1)
ma = np.array([1])        # MA coefficient (no moving average part)

# Generate a stationary, non-white-noise time series
arma_process = ArmaProcess(ar, ma)
n_samples = 1460  # 4 years worth of daily data

stationary_series = arma_process.generate_sample(nsample=n_samples)

# Create a DataFrame with this series
dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
stationary_ts_df = pd.DataFrame({'Date': dates, 'Value': stationary_series})
stationary_ts_df.set_index('Date', inplace=True)

# Plot the series
plt.figure(figsize=(12, 6))
plt.plot(stationary_ts_df['Value'])
plt.title("Stationary Series with Structure (2020 - 2023)")
plt.xlabel("Date")
plt.ylabel("Value")
plt.show()

# Display the first few values
stationary_ts_df

# %% [markdown]
# ### Finding lags

# %%
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plot_acf(stationary_ts_df)
plot_pacf(stationary_ts_df)
plt.show()

# %% [markdown]
# ### White noise check

# %%
from statsmodels.stats.diagnostic import acorr_ljungbox
acorr_ljungbox(x=stationary_ts_df, lags=[30])

# %%
2.786094e-292 < 0.05

# %% [markdown]
# *Conclusion*: The `stationary_ts_df` is not a white noise

# %% [markdown]
# ### Random walk test

# %%
from arch.unitroot import VarianceRatio

stat_vr = VarianceRatio(stationary_ts_df)

print("The variance ration statistic:", stat_vr.stat)
print("The variance ratio p-val:", stat_vr.pvalue)

# %% [markdown]
# *Conclusion*: Since pvalue is 6.148723752374963e-10, less than 0.05, so it's not a random walk.

# %% [markdown]
# ### Seasonal decompose of `stationary_ts_df`

# %%
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
seasonal_decompose(stationary_ts_df).plot()
plt.show()

# %%
seasonal_wave = seasonal_decompose(stationary_ts_df).seasonal

plt.plot(seasonal_wave)

# %%
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plot_acf(seasonal_wave)
plot_pacf(seasonal_wave)
plt.show()

# %% [markdown]
# ## AR(AutoRegression) Model

# %% [markdown]
# ### Train test split

# %%
from pandas import to_datetime

split_date = to_datetime('2023-01-01')
train_ts = stationary_ts_df.loc[stationary_ts_df.index <= split_date]
test_ts = stationary_ts_df.loc[stationary_ts_df.index > split_date]

# %%
train_ts

# %%
test_ts

# %% [markdown]
# ### Modeling

# %%
import matplotlib.pyplot as plt
from numpy import sqrt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AutoReg
from warnings import filterwarnings

filterwarnings('ignore')

model = AutoReg(endog=train_ts['Value'], lags=15).fit()
predictions = model.predict(start = len(train_ts), end=len(train_ts)+len(test_ts)-1, dynamic=False)

rmse = sqrt(mean_squared_error(test_ts, predictions))

plt.figure(figsize=(10,5))
plt.plot(test_ts.index, test_ts['Value'], color='#00B9E3', linestyle='-', label='Actual value')
plt.plot(test_ts.index, predictions.values, color='#F8766D', linestyle='--', label='Predictions')
plt.title('Actual vs Predicted Data')
plt.legend()
plt.show()

rmse, stationary_ts_df['Value'].mean(), predictions.mean()

# %% [markdown]
# ## MA model

# %% [markdown]
# ### ARIMA

# %%
import matplotlib.pyplot as plt
from numpy import sqrt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(train_ts, order = (22,0,40)).fit()
predictions = model.predict(start = len(train_ts), end=len(train_ts)+len(test_ts)-1, dynamic=False)

rmse = sqrt(mean_squared_error(test_ts, predictions))

plt.figure(figsize=(10,5))
plt.plot(test_ts.index, test_ts['Value'], color='#00B9E3', linestyle='-', label='Actual value')
plt.plot(test_ts.index, predictions.values, color='#F8766D', linestyle='--', label='Predictions')
plt.title('Actual vs Predicted Data')
plt.legend()
plt.show()

rmse, stationary_ts_df['Value'].mean(), predictions.mean()

# %% [markdown]
# ### SARIMA

# %%
seasonality_df = seasonal_wave.to_frame()
seasonality_df

# %%
seasonality_df.diff(160).dropna()

# %%
adf_stationarity_test(seasonality_df.diff(+3).dropna())

# %%
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from numpy import sqrt
from sklearn.metrics import mean_squared_error

p,d,q = 2,0,2
P,D,Q = 1,1,1
s = 12

model = SARIMAX(train_ts, order = (p,d,q), seasonal_order = (P,D,Q,s)).fit()
predictions = model.predict(start = len(train_ts), end=len(train_ts)+len(test_ts)-1, dynamic=False)

rmse = sqrt(mean_squared_error(test_ts, predictions))

plt.figure(figsize=(10,5))
plt.plot(test_ts.index, test_ts['Value'], color='#00B9E3', linestyle='-', label='Actual value')
plt.plot(test_ts.index, predictions.values, color='#F8766D', linestyle='--', label='Predictions')
plt.title('Actual vs Predicted Data')
plt.legend()
plt.show()

rmse, stationary_ts_df['Value'].mean(), predictions.mean()

# %%
model.summary()

# %% [markdown]
# # Multi-variate models

# %% [markdown]
# ## Getting stock data

# %%
from yfinance import download

apple_stock_23 = download('AAPL', start='2023-01-01')
# tesla_stock_23 = download('TSLA', start='2023-01-01')

# %%
apple_stock_23

# %% [markdown]
# ## Making multi-variate model

# %% [markdown]
# ### Data pre-processing

# %%
from statsmodels.tsa.stattools import grangercausalitytests
from pandas import DataFrame

def granger_causality_matrix(data, maxlag):
    variables = data.columns
    granger_matrix = DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)

    for col in variables:
        for row in variables:
            if row != col:
                test_result = grangercausalitytests(data[[row, col]], maxlag=maxlag, verbose=False)
                # Store p-value for the smallest lag where causality exists
                p_values = [round(test_result[i+1][0]['ssr_ftest'][1], 4) for i in range(maxlag)]
                granger_matrix.loc[row, col] = np.min(p_values)
            else:
                granger_matrix.loc[row, col] = np.nan  # Causality with itself is not relevant

    return granger_matrix

# %%
granger_causality_matrix(data=apple_stock_23, maxlag=5)

# %%
adf_stationarity_test(apple_stock_23['Volume'])

# %%
adf_stationarity_test(apple_stock_23['Open'])

# %% [markdown]
# ## Train test split

# %%
# from pandas import to_datetime
split_date = '2024-01-23'
train_apple_23 = apple_stock_23[['Open', 'Volume', 'Close']].loc[apple_stock_23.index <= split_date]
test_apple_23 = apple_stock_23[['Open', 'Volume', 'Close']].loc[apple_stock_23.index > split_date]

# %%
train_apple_23

# %%
test_apple_23

# %% [markdown]
# ## VAR & VARMAX

# %%
from statsmodels.tsa.api import VAR
from numpy import sort

model = VAR(train_apple_23)
model.select_order(maxlags=10).summary()
model_fitted = model.fit(maxlags=6)
forecast_steps = len(test_apple_23)
predictions = model_fitted.forecast(y=test_apple_23.values, steps=forecast_steps)
predictions = sort(predictions, axis=0)
# test_apple_23['close_prediction']
predictions_df = DataFrame(data=predictions, columns=test_apple_23.columns, index = test_apple_23.index)
predictions_df

# %%
predictions_df.columns = ['Open', 'Volume', 'Close']
predictions_df

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(apple_stock_23.index, apple_stock_23['Close'], label='Actual Close')
plt.plot(predictions_df.index, predictions_df['Close'], label = 'Predicted Close')
plt.title('Actual Close Vs Predicted Close of AAPL')
plt.legend()
plt.show()

# %%
from statsmodels.tsa.statespace.varmax import VARMAX

varmax_model = VARMAX(train_apple_23.diff(+30).dropna(), order=(6,0), enforce_stationarity=True).fit(disp=False)
varmax_model.summary()

# %%
n_steps = 14
predict = varmax_model.get_prediction(start=len(train_apple_23),end=len(train_apple_23) + n_steps-1)

varmax_predictions=predict.predicted_mean
varmax_predictions

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(13,16))

plt.subplot(211)
plt.plot(apple_stock_23.index, apple_stock_23['Close'], label='Actual Close')
plt.title('Actual Close of AAPL')
plt.legend()
plt.show()

plt.subplot(212)
plt.plot(varmax_predictions.index, varmax_predictions['Close_AAPL'], label = 'Predicted Close')
plt.title('Predicted Close of AAPL')
plt.legend()
plt.show()

plt.tight_layout()

# %% [markdown]
# > *Since stock data is always full of irregularities, so even after differencing and all stationarizing seemingly fails because there're underlying cycles, trends which cannot be removed, even if you does, the data would be white-noise or semi white-noise. These underlying cycles and trends are important for stock data analysis.*
# 
# > ***One thing to keep in mind, the trends, seasonality, non-linear cycles of certain occurrences are important to signify the trends, either pump-or-dump, bullish/bearish, etc.***

# %% [markdown]
# # Smoothing technique

# %% [markdown]
# Smoothing data involves reducing the noise or irregularities in a dataset to reveal underlying trends or patterns. This technique is useful when dealing with noisy or erratic data, as it allows for easier interpretation of the data.
# 
# There are several methods for smoothing data in Python, including moving averages, Savitzky-Golay filters, and exponential smoothing. Each method has its strengths and weaknesses and can be applied to different types of datasets.
# 
# ### Data smoothing
# Data smoothing is a technique used to remove noise or irregularities from a dataset. It involves creating a new dataset that represents the original data in a smoother way. The main objective of data smoothing is to identify patterns or trends in the data by reducing the noise or random fluctuations that can obscure them.
# 
# There are several methods available for data smoothing such as moving average, Savitzky-Golay filter, and exponential smoothing. Each method has its own advantages and disadvantages depending on the type of data and the desired level of smoothing.
# 
# The moving average method calculates the average of a set of values over a specified window size. This method is useful for removing high-frequency noise from the data but may not be effective for removing low-frequency noise.
# 
# The Savitzky-Golay filter is a polynomial smoothing technique that fits a polynomial function to a subset of adjacent data points. This method is effective for removing both high-frequency and low-frequency noise from the data.
# 
# Exponential smoothing is another popular technique used for time series analysis. It involves assigning weights to past observations in such a way that more recent observations are given greater weight than older ones. This method is particularly useful for forecasting future values based on past trends.
# 
# Overall, data smoothing can be a powerful tool for analyzing datasets and identifying underlying patterns or trends. By choosing the appropriate smoothing method for your data, you can improve its accuracy and usefulness for further analysis.
# 
# ### Rationale of data smoothing
# Data smoothing is an essential technique in data analysis that helps to remove noise from data. When working with large datasets, it’s common to have some irregularities or noise that can obscure important trends or patterns in the data. Smoothing techniques help to eliminate these irregularities and provide a clearer picture of the underlying patterns.
# 
# Data smoothing is particularly important when dealing with time-series data, where there may be many fluctuations and sudden changes over time. By applying smoothing techniques, analysts can better understand the long-term trends in the data and make more accurate predictions about future behavior.
# 
# Furthermore, smoothed data can be easier to interpret and communicate to others. By removing noise and highlighting important trends, smoothed data can help to tell a more compelling story about the insights that can be drawn from the data.
# 
# Overall, data smoothing is an essential tool for any analyst who wants to gain deeper insights into their data and make more informed decisions based on that information.

# %% [markdown]
# ### Types of Data Smoothing Techniques
# Data smoothing is a technique used to remove noise from a data set, allowing for easier identification of trends and patterns. There are several types of data smoothing techniques, each with its own strengths and weaknesses. In this section, we will explore the most common types of data smoothing techniques used in Python.
# 
# #### Moving Average
# The moving average technique involves calculating the average of a subset of data points within a specified window size. The window size determines how many data points are included in the calculation. Moving averages are useful for identifying trends in data sets as they smooth out fluctuations in the data.
# 
# Here is an example of how to implement moving averages in Python:

# %%
import matplotlib.pyplot as plt

# Using apple stock data from 2023
plt.figure(figsize=(12,7))
plt.plot(apple_stock_23.index, apple_stock_23['Close'], linestyle='-', color='#1124F6', label = 'Original close')
plt.plot(apple_stock_23['Close'].rolling(window=15).mean(), linestyle='-', color='#FF1126', label='MA smoothed close')
plt.title('Apple Close Original Vs MA series')
plt.legend()
plt.show()

# %% [markdown]
# #### Simple smoothing
# Holt-Winters Simple Exponential Smoothing (often referred to as single smoothing) is a forecasting method for time series data that assumes no trend or seasonality in the data. It uses a weighted average of past observations, where more recent observations are given higher weights, controlled by a smoothing parameter \( \alpha \) (0 ≤ \( \alpha \) ≤ 1).
# 
# The formula for updating the level at time \( t \) is:
# 
# \[
# L_t = \alpha Y_t + (1 - \alpha) L_{t-1}
# \]
# 
# - **\( L_t \)**: Smoothed value (level) at time \( t \)
# - **\( Y_t \)**: Observed value at time \( t \)
# - **\( \alpha \)**: Smoothing parameter
# 
# It is ideal for data without trends or seasonality and helps reduce noise while capturing the overall pattern of the series.

# %%
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import matplotlib.pyplot as plt

# Using apple stock data from 2023
simple_exp = SimpleExpSmoothing(endog=apple_stock_23['Close']['AAPL']).fit(smoothing_level=0.7)
predictions_smooth = simple_exp.fittedvalues

plt.figure(figsize=(12,7))
plt.plot(apple_stock_23.index, apple_stock_23['Close'], linestyle='-', color='#1F24F6', label = 'Original close')
plt.plot(predictions_smooth, linestyle='-', color='#FF1126', label='Smooth Exponential Close')
plt.title('Apple Close Original Vs Smooth Exponential series')
plt.legend()
plt.show()

# %% [markdown]
# ##### Prediction using simple exponential smooth

# %%
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import matplotlib.pyplot as plt

simple_exp_model = SimpleExpSmoothing(endog=train_apple_23['Close']['AAPL']).fit(smoothing_level=0.7)
prediction_simp_exp = simple_exp.predict(start = len(train_apple_23['Close']), end = len(train_apple_23) + len(test_apple_23) - 1)
prediction_simp_exp = prediction_simp_exp.to_frame()
# prediction_simp_exp = prediction_simp_exp[prediction_simp_exp.index == test_apple_23.index]
# prediction_simp_exp.index = apple_stock_23.index
prediction_simp_exp.columns = ['simple_exp_predict']

plt.figure(figsize=(12,7))
plt.plot(test_apple_23.index, test_apple_23['Close'], linestyle='-', color='#1F24F6', label = 'Test close')
plt.plot(prediction_simp_exp, linestyle='-', color='#FF1126', label='Smooth Exponential Predicted Close')
plt.title('Apple Close Original Vs Smooth Exponential Predicted Series')
plt.legend()
plt.show()

prediction_simp_exp

# %%
test_apple_23['Close'].mean()

# %%
prediction_simp_exp['simple_exp_predict'].mean()

# %% [markdown]
# #### Exponential Smoothing
# Exponential smoothing is a popular technique for time series forecasting. It involves assigning exponentially decreasing weights to older observations, with more recent observations receiving higher weights. This technique is useful for capturing trends and seasonality in time series data.
# 
# Here is an example of how to implement exponential smoothing in Python using the Holt-Winters method:

# %%
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

# Using apple stock data from 2023
exponential_model = ExponentialSmoothing(endog=apple_stock_23['Close'], trend='mul', seasonal='mul', seasonal_periods=7).fit()
predictions = exponential_model.predict(start=0, end=len(apple_stock_23)-1)

plt.figure(figsize=(12,7))
plt.plot(apple_stock_23.index, apple_stock_23['Close'], linestyle='-', color='#1F24F6', label = 'Original close')
plt.plot(predictions, linestyle='-', color='#FF1126', label='Exponential Smooth Close')
plt.title('Apple Close Original Vs EMA series')
plt.legend()
plt.show()

# %% [markdown]
# ##### Exponential smoothing for forecasting

# %%
train_exp_model = ExponentialSmoothing(endog=train_apple_23['Close'], trend='mul', seasonal='mul', seasonal_periods=7).fit()
forecast_close = train_exp_model.predict(start=len(train_apple_23), end=len(train_apple_23)+len(test_apple_23)-1)
predictions = predictions.to_frame()
predictions = predictions[predictions.index > split_date]
predictions.columns = ['Predicted Close']
predictions.index = test_apple_23.index

plt.figure(figsize=(12,7))
plt.plot(test_apple_23.index, test_apple_23['Close'], linestyle='-', color='#FF1126', label = 'test close')
plt.plot(predictions.index, predictions['Predicted Close'], linestyle='-', color='#1F24F6', label='Exponential Smooth Predicted Close')
plt.title('Apple Close Original Vs EMA Predicted series')
plt.legend()
plt.show()

# %%
predictions

# %%
test_apple_23['Close'].mean()

# %%
predictions['Predicted Close'].mean()

# %% [markdown]
# #### Lowess Smoothing
# 
# Lowess smoothing, short for locally weighted scatterplot smoothing, is a non-parametric technique for data smoothing. It involves fitting a regression line to a subset of the data using weighted least squares, with the weights determined by the distance between each point and the point being estimated. This technique is useful for identifying trends and patterns in noisy data.
# 
# Here is an example of how to implement lowess smoothing in Python:

# %%
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt

# Extracting the 'Close' column as a 1D array
lowess_close = lowess(endog=train_apple_23['Close']['AAPL'].values, exog=range(len(train_apple_23['Close']['AAPL'])), frac=0.4)

# Plotting
plt.figure(figsize=(12, 7))
plt.plot(apple_stock_23.index, apple_stock_23['Close'], linestyle='-', color='#FF1126', label='Original Close')
plt.plot(lowess_close[:, 1], linestyle='-', color='#1F24F6', label='LOWESS Smoothed')
plt.title('Apple Close - Original vs LOWESS Smoothed')
plt.legend()
plt.show()

# %% [markdown]
# #### Kalman Filtering
# 
# Kalman filtering is a recursive algorithm that uses a series of measurements observed over time to estimate unknown variables. It involves predicting the state of a system at time t based on the state at time t-1 and then updating the prediction based on new measurements. This technique is useful for dealing with noisy data and can be used for both linear and non-linear systems.
# 
# Here is an example of how to implement Kalman filtering in Python:

# %%
import matplotlib.pyplot as plt
from pykalman import KalmanFilter

kalman_filter = KalmanFilter(
    transition_matrices=[1],
    observation_matrices=[1],
    initial_state_mean=apple_stock_23['Close']['AAPL'].iloc[0],
    initial_state_covariance=1,
    observation_covariance=1,
    transition_covariance=0.01
)

state_means, _ = kalman_filter.filter(apple_stock_23['Close']['AAPL'])
state_means = state_means.flatten()

plt.figure(figsize=(12, 7))
plt.plot(apple_stock_23.index, apple_stock_23['Close'], linestyle='-', color='#FF1126', label='Original Close')
plt.plot(state_means, linestyle='-', color='#1F24F6', label='Kalman filter state means Smoothed')
plt.title('Apple Close - Original vs KalmanFilter Smoothed')
plt.legend(loc= 'upper center')
plt.show()

# %% [markdown]
# ##### Using `filterpy`

# %%

from filterpy.kalman import KalmanFilter
import numpy as np

# Define state transition matrix
F = np.array([[1, 1], [0, 1]])

# Define measurement matrix
H = np.array([[1, 0]])

# Define process noise covariance matrix
Q = np.array([[0.1, 0], [0, 0.01]])

# Define measurement noise covariance matrix
R = np.array([[1]])

# Create Kalman filter object
kf = KalmanFilter(dim_x=2, dim_z=1)
kf.F = F
kf.H = H
kf.Q = Q
kf.R = R

# Initialize state vector and covariance matrix
x0 = np.array([0, 0])
P0 = np.eye(2) * 1000
kf.x = x0
kf.P = P0

# Generate measurements
measurements = [1, 2, 3, 4, 5]

# Perform Kalman filtering
filtered_states = []
for z in measurements:
    kf.predict()
    kf.update(z)
    filtered_states.append(kf.x)

print(filtered_states)

# %% [markdown]
# #### Savitzky-Golay Filtering
# 
# Savitzky-Golay filtering is a technique for smoothing noisy data that involves fitting a polynomial to a moving window of data points and then using the coefficients of the polynomial to estimate the smoothed values. This technique is useful for preserving the shape of the data while removing noise.
# 
# Here is an example of how to implement Savitzky-Golay filtering in Python:

# %%
from scipy.signal import savgol_filter

savgol_filtered = savgol_filter(x=apple_stock_23['Close']['AAPL'], window_length=7, polyorder=6)

plt.figure(figsize=(12, 7))
plt.plot(apple_stock_23.index, apple_stock_23['Close'], linestyle='-', color='#FF1126', label='Original Close')
plt.plot(savgol_filtered, linestyle='-', color='#1F24F6', label='Savitzky-Golay Smoothed')
plt.title('Apple Close - Original vs Savitzky-Golay Smoothed')
plt.legend(loc= 'upper center')
plt.show()

# %% [markdown]
# # AIC and BIC of forecasting models
# For the demo, we'll use only VARMAX fitted model
# 
# The lesser they're, the better the model is gonna be.

# %%
varmax_model.aic, varmax_model.bic

# %% [markdown]
# # **Data Pre-processing for Time Series Analysis**
# 
# ## **1. Handling Missing Values**
# Missing data in a time series can occur in two ways:
# - **Cases**:
#   - A date with `null`/`NaN` value: The date is present, but the corresponding value is missing.
#   - A date is not present: A time gap where certain dates are entirely absent from the dataset.
# 
# - **Methods to Handle Missing Values**:
#   - **Imputation**:
#     - Replace missing values with statistical measures like **mean**, **median**, or **mode**.
#     - Use **forward-fill** (copy the last available value) or **backward-fill** (copy the next available value).
#     - Domain-specific constants (e.g., zero in some financial datasets).
#   - **Interpolation**:
#     - **Linear Interpolation**: Assumes linear progression between missing values.
#     - **Spline Interpolation**: Fits a spline curve for smoother interpolation.
#     - **Polynomial Interpolation**: Uses higher-degree polynomials to approximate the missing values.
#   - **Predictive Modeling**:
#     - Train machine learning models (**regression**, **k-nearest neighbors**, **clustering**) to predict missing values based on available data.
#     - Utilize specialized time series forecasting methods like ARIMA to infer missing points.
# 
# ## **2. Making Data Stationary & Handling Outliers**
# Stationarity is essential for many time series models like ARIMA and Exponential Smoothing. Outliers can disrupt the signal and model performance.
# 
# - **Methods to Ensure Stationarity**:
#   - **Moving Average (MA)**: Smoothens the series by averaging over a fixed window, reducing noise and seasonal fluctuations.
#   - **Smoothing**:
#     - **Exponential Moving Average (EMA)** for weighted smoothing.
#     - Seasonal decomposition techniques (e.g., STL) to extract trends and seasonality.
#   - **Transformations**:
#     - **Log Transformation**: Compresses the scale to reduce variance.
#     - **Square Root Transformation**: Reduces extreme spikes.
#     - **Box-Cox Transformation**: Flexible transformation that stabilizes variance and normalizes data.
# 
# - **Handling Outliers**:
#   - Identify outliers using thresholds like the **IQR rule**, **Z-score**, or domain knowledge.
#   - Replace outliers with interpolated values, rolling averages, or impute them using models.
# 
# ## **3. Resampling**
# Resampling adjusts the frequency or granularity of a time series to better match the analysis needs.
# 
# - **Types of Resampling**:
#   - **Up-sampling**:
#     - Increase the frequency of the data (e.g., daily to hourly).
#     - Fill gaps using interpolation or imputation techniques.
#   - **Down-sampling**:
#     - Reduce the frequency of the data (e.g., daily to weekly).
#     - Aggregate data using statistical measures (e.g., mean, sum, max).
# 
# - **Use Cases**:
#   - Resampling for aligning datasets with different time frequencies.
#   - Converting raw data to meaningful intervals for seasonal or trend analysis.
# 
# ## **4. Additional Pre-processing Steps**
# - **Encoding Cyclic Features**: For datetime features like month, day, or hour, use sine-cosine encoding to capture cyclical patterns.
# - **Normalization/Scaling**: Standardize data (e.g., MinMaxScaler, StandardScaler) for models sensitive to scale, like neural networks.
# - **Seasonal Decomposition**: Break the series into trend, seasonality, and residual components to better understand its structure.
# 
# ## **Pre-processing Pipeline for Time Series**
# 1. Load the data and ensure proper **datetime indexing**.
# 2. Handle missing values using appropriate techniques.
# 3. Perform **stationarity tests** (ADF test, KPSS test) and transformations if necessary.
# 4. Detect and handle outliers.
# 5. Resample data to match the desired frequency.
# 6. Normalize or scale data if required.
# 7. Decompose the series if needed to separate trends and seasonality.
# 
# This structured approach ensures the data is clean, consistent, and ready for modeling or analysis.
# 
# ---
# Here's the code demo of each concept

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import boxcox
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# Create synthetic time series data
np.random.seed(42)
date_rng = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
data = np.random.randint(50, 100, size=(len(date_rng))) + np.sin(np.linspace(0, 3.14 * 2, len(date_rng))) * 20

# Introduce missing values and outliers
data[50] = np.nan  # Missing value
data[100] = 500    # Outlier
data[200:205] = np.nan  # Multiple missing values

time_series = pd.DataFrame({'Date': date_rng, 'Value': data})
time_series.set_index('Date', inplace=True)

# Step 1: Handle missing values
# Imputation using forward-fill
time_series['Value_ffill'] = time_series['Value'].fillna(method='ffill')

# Imputation using linear interpolation
time_series['Value_interpolated'] = time_series['Value'].interpolate(method='linear')

# Step 2: Handle outliers
# Replace outliers with the median of the series
threshold = 3  # Example threshold for outlier detection using z-score
median_value = time_series['Value_interpolated'].median()
time_series['Value_outlier_removed'] = np.where((time_series['Value_interpolated'] - median_value).abs() > threshold * median_value,
                                                median_value,
                                                time_series['Value_interpolated'])

# Step 3: Apply Box-Cox Transformation
# Ensure no zeros or negative values (Box-Cox requires strictly positive data)
time_series['Value_positive'] = time_series['Value_outlier_removed'] + abs(time_series['Value_outlier_removed'].min()) + 1
time_series['Value_boxcox'], boxcox_lambda = boxcox(time_series['Value_positive'])

# Step 4: Seasonal decomposition
# Decompose the Box-Cox transformed series
decomposition = seasonal_decompose(time_series['Value_boxcox'], model='additive', period=30)
time_series['Trend'] = decomposition.trend
time_series['Seasonal'] = decomposition.seasonal
time_series['Residual'] = decomposition.resid

# Step 5: Stationarity Test
# Perform ADF Test on Box-Cox transformed data
result = adfuller(time_series['Value_boxcox'].dropna())
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")
print(f"Box-Cox Lambda: {boxcox_lambda}")

# Step 6: Plot Results
plt.figure(figsize=(14, 10))

# Original series
plt.subplot(3, 2, 1)
plt.plot(time_series.index, time_series['Value'], label='Original')
plt.title('Original Series')
plt.legend()

# Missing value handling
plt.subplot(3, 2, 2)
plt.plot(time_series.index, time_series['Value_ffill'], label='Forward-Filled')
plt.plot(time_series.index, time_series['Value_interpolated'], label='Interpolated', linestyle='--')
plt.title('Missing Value Handling')
plt.legend()

# Outlier handling
plt.subplot(3, 2, 3)
plt.plot(time_series.index, time_series['Value_interpolated'], label='Before Outlier Removal')
plt.plot(time_series.index, time_series['Value_outlier_removed'], label='After Outlier Removal', linestyle='--')
plt.title('Outlier Handling')
plt.legend()

# Box-Cox transformation
plt.subplot(3, 2, 4)
plt.plot(time_series.index, time_series['Value_boxcox'], label='Box-Cox Transformed', color='purple')
plt.title('Box-Cox Transformation')
plt.legend()

# Decomposed components
plt.subplot(3, 2, 5)
plt.plot(time_series.index, time_series['Trend'], label='Trend', color='orange')
plt.plot(time_series.index, time_series['Seasonal'], label='Seasonal', color='green')
plt.title('Decomposed Trend and Seasonality')
plt.legend()

# Residual component
plt.subplot(3, 2, 6)
plt.plot(time_series.index, time_series['Residual'], label='Residual', color='red')
plt.title('Residual Component')
plt.legend()

plt.tight_layout()
plt.show()

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# Create synthetic time series data
np.random.seed(42)
date_rng = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
data = np.random.randint(50, 100, size=(len(date_rng))) + np.sin(np.linspace(0, 3.14 * 2, len(date_rng))) * 20

# Introduce missing values and outliers
data[50] = np.nan  # Missing value
data[100] = 500    # Outlier
data[200:205] = np.nan  # Multiple missing values

time_series = pd.DataFrame({'Date': date_rng, 'Value': data})
time_series.set_index('Date', inplace=True)

# Step 1: Handle missing values
# Imputation using forward-fill
time_series['Value_ffill'] = time_series['Value'].fillna(method='ffill')

# Imputation using linear interpolation
time_series['Value_interpolated'] = time_series['Value'].interpolate(method='linear')

# Step 2: Handle outliers
# Replace outliers with the median of the series
threshold = 3  # Example threshold for outlier detection using z-score
median_value = time_series['Value_interpolated'].median()
time_series['Value_outlier_removed'] = np.where((time_series['Value_interpolated'] - median_value).abs() > threshold * median_value,
                                                median_value,
                                                time_series['Value_interpolated'])

# Step 3: Make data stationary (apply log transformation)
time_series['Value_log'] = np.log1p(time_series['Value_outlier_removed'])

# Step 4: Seasonal decomposition
# Decompose the log-transformed series
decomposition = seasonal_decompose(time_series['Value_log'].dropna(), model='additive', period=30)
time_series['Trend'] = decomposition.trend
time_series['Seasonal'] = decomposition.seasonal
time_series['Residual'] = decomposition.resid

# Step 5: Stationarity Test
# Perform ADF Test
result = adfuller(time_series['Value_log'].dropna())
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")

# Step 6: Plot Results
plt.figure(figsize=(14, 10))

# Original series
plt.subplot(3, 2, 1)
plt.plot(time_series.index, time_series['Value'], label='Original')
plt.title('Original Series')
plt.legend()

# Missing value handling
plt.subplot(3, 2, 2)
plt.plot(time_series.index, time_series['Value_ffill'], label='Forward-Filled')
plt.plot(time_series.index, time_series['Value_interpolated'], label='Interpolated', linestyle='--')
plt.title('Missing Value Handling')
plt.legend()

# Outlier handling
plt.subplot(3, 2, 3)
plt.plot(time_series.index, time_series['Value_interpolated'], label='Before Outlier Removal')
plt.plot(time_series.index, time_series['Value_outlier_removed'], label='After Outlier Removal', linestyle='--')
plt.title('Outlier Handling')
plt.legend()

# Log transformation
plt.subplot(3, 2, 4)
plt.plot(time_series.index, time_series['Value_log'], label='Log-Transformed', color='purple')
plt.title('Log Transformation')
plt.legend()

# Decomposed components
plt.subplot(3, 2, 5)
plt.plot(time_series.index, time_series['Trend'], label='Trend', color='orange')
plt.plot(time_series.index, time_series['Seasonal'], label='Seasonal', color='green')
plt.title('Decomposed Trend and Seasonality')
plt.legend()

# Residual component
plt.subplot(3, 2, 6)
plt.plot(time_series.index, time_series['Residual'], label='Residual', color='red')
plt.title('Residual Component')
plt.legend()

plt.tight_layout()
plt.show()

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Generate a synthetic time series dataset
np.random.seed(42)
date_rng = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')  # Daily frequency
data = np.random.randint(50, 100, size=(len(date_rng)))

time_series = pd.DataFrame({'Date': date_rng, 'Value': data})
time_series.set_index('Date', inplace=True)

# Plot the original time series
plt.figure(figsize=(12, 6))
plt.plot(time_series.index, time_series['Value'], label='Original (Daily)', color='blue')
plt.title('Original Time Series (Daily Frequency)')
plt.legend()
plt.show()

# **Down-sampling**: Convert daily data to monthly data (mean of each month)
monthly_resampled = time_series.resample('M').mean()

# **Up-sampling**: Convert daily data to hourly data (forward-fill to fill missing values)
hourly_resampled = time_series.resample('H').ffill()

# Plotting resampled data
plt.figure(figsize=(12, 10))

# Down-sampled data
plt.subplot(2, 1, 1)
plt.plot(monthly_resampled.index, monthly_resampled['Value'], label='Down-Sampled (Monthly)', color='green')
plt.title('Down-Sampled Time Series (Monthly Frequency)')
plt.legend()

# Up-sampled data
plt.subplot(2, 1, 2)
plt.plot(hourly_resampled.index[:500], hourly_resampled['Value'][:500], label='Up-Sampled (Hourly)', color='orange')
plt.title('Up-Sampled Time Series (Hourly Frequency - First 500 Hours)')
plt.legend()

plt.tight_layout()
plt.show()

# Display summary statistics
print("Original Time Series:")
print(time_series.head())
print("\nDown-Sampled Time Series (Monthly):")
print(monthly_resampled.head())
print("\nUp-Sampled Time Series (Hourly):")
print(hourly_resampled.head())

# %% [markdown]
# | Platform     | Profile Link                                   |
# |--------------|-----------------------------------------------|
# | LinkedIn     | [Fawad Awan](https://linkedin.com/in//fawad-awan-893a58171) |
# | GitHub       | [JackTheProgrammer](https://github.com/JackTheProgrammer)      |
#