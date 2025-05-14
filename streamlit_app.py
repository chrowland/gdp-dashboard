import streamlit as st
import statsmodels.api as sm
import pandas as pd
import numpy as np
import altair as alt
import datetime

# Set page configuration
st.set_page_config(page_title="Time Series Seasonality Simulator", layout="wide")

st.title("📈📈📈📈📈📈 Time Series Seasonality Simulator 📈📈📈📈📈📈")

# Sidebar for user inputs
st.sidebar.header("Configure Time Series Components")

# Trend selection
trend_type = st.sidebar.selectbox(
    "Select Trend Type",
    options=["Linear", "Exponential", "Quadratic"]
)

intercept = st.sidebar.slider("Intercept", min_value=30.0, max_value=100.0, value=30.0, step=1.0)

# Trend parameters
if trend_type == "Linear":
    slope = st.sidebar.slider("Slope", min_value=0.0, max_value=5.0, value=0.2, step=0.1)
    #intercept = st.sidebar.slider("Intercept", min_value=5.0, max_value=50.0, value=10.0, step=1.0)
elif trend_type == "Exponential":
    #intercept = st.sidebar.slider("Intercept", min_value=5.0, max_value=50.0, value=10.0, step=1.0)
    base = st.sidebar.slider("Base", min_value=0.1, max_value=5.0, value=1.1, step=0.1)
    rate = st.sidebar.slider("Rate (r)", min_value=0.01, max_value=0.15, value=0.05, step=0.005)
    st.sidebar.subheader("$trend = int + base^r$")
elif trend_type == "Quadratic":
    #intercept = st.sidebar.slider("Intercept", min_value=5.0, max_value=50.0, value=10.0, step=1.0)
    a = st.sidebar.slider("Coefficient a", min_value=0.0, max_value=2.0, value=0.01, step=0.01)
    b = st.sidebar.slider("Coefficient b", min_value=0.0, max_value=10.0, value=0.5, step=0.1)
    st.sidebar.subheader("$trend = int + a*X^2 +b*X$")

# Cycle parameters
st.sidebar.subheader("Cycle Component")
cycle_amplitude = st.sidebar.slider("Cycle Amplitude", min_value=0.0, max_value=10.0, value=6.7, step=0.1)
cycle_period = st.sidebar.slider("Cycle Period (months)", min_value=2, max_value=60, value=43, step=1)
cycle_phase = st.sidebar.slider("Cycle Phase (radians)", min_value=0.0, max_value=2*np.pi, value=1.5, step=0.1)

# Seasonality parameters
st.sidebar.subheader("Seasonality Component")
seasonality_amplitude = st.sidebar.slider("Seasonality Amplitude", min_value=0.0, max_value=10.0, value=6.0, step=0.1)
seasonality_mult = st.sidebar.slider("Seasonality Multiplier (0 is Additive)", min_value=0.0, max_value=3.0, value=0.0, step=1.0)

# Noise parameters
st.sidebar.subheader("Noise Component")
noise_std = st.sidebar.slider("Noise Standard Deviation", min_value=0.0, max_value=5.0, value=2.0, step=0.1)

# Parameters for outliers
outliers=st.sidebar.checkbox("Outliers")









# Generate date range
start_date = datetime.date(2018, 1, 1)
end_date = datetime.date.today()
dates = pd.date_range(start=start_date, end=end_date, freq='MS')
n = len(dates)
t = np.arange(n)

# Generate components
# Trend
if trend_type == "Linear":
    trend = intercept + slope * t
elif trend_type == "Exponential":
    trend = intercept + base * np.exp(rate * t)
elif trend_type == "Quadratic":
    trend = intercept + a * t**2 + b * t

# Cycle
cycle = cycle_amplitude * np.sin(2 * np.pi * t / cycle_period + cycle_phase)

# Trend + Cycle
trend_cycle = trend + cycle

# Seasonality
seasonality = seasonality_amplitude* (1+seasonality_mult*np.log(trend_cycle/intercept))* np.sin(2 * np.pi * t / 12)

# Noise
np.random.seed(42)  # For reproducibility
noise = np.random.normal(0, noise_std, n)
if outliers:
    outlier_count=int(n/10)
    outlier_series=np.array([1]*(outlier_count)+[-1]*(outlier_count)+[0]*(n-2*outlier_count))
    np.random.seed(42)
    np.random.shuffle(outlier_series)
    noise=noise+outlier_series*noise_std*4

# Trend + Noise
trend_noise = trend+noise

# Seasonality + Cycle
season_cycle = seasonality + cycle

# Composite time series
composite = trend + cycle + seasonality + noise

# Create DataFrame
df = pd.DataFrame({
    "Date": dates,
    "Composite": composite,
    "Trend": trend,
    "Cycle": cycle,
    "Seasonality": seasonality,
    "Noise": noise,
    "Season_cycle": season_cycle,
    "Trend_noise": trend_noise,
    "Trend_cycle": trend_cycle
})
df.set_index("Date", inplace=True)

# Display composite time series
st.subheader("Composite Time Series")
st.line_chart(df["Composite"])

# Display individual components
st.subheader("🔍 Individual Components")

col1, col2 = st.columns(2)

with col1:
    st.write("**Trend Component**")
    st.line_chart(df["Trend"])

    st.write("**Noise Component**")
    st.line_chart(df["Noise"])

    st.write("**Trend and Noise**")
    st.line_chart(df["Trend_noise"])
                  
with col2:
    st.write("**Cycle Component**")
    st.line_chart(df["Cycle"])

    st.write("**Seasonality Component**")
    st.line_chart(df["Seasonality"])

    st.write("**Seasonality and Cycle**")
    st.line_chart(df["Season_cycle"])

Recipe=df
st.download_button("Download Timeseries Recipe",Recipe.to_csv(),"TSRecipe.csv",use_container_width=True)

from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
st.header("🔍 Time Series Decomposition Analysis")

# 1) Train/Test Split Slider
st.subheader("1️⃣ Select Train/Test Split Point")
split_index = st.slider(
    "Select the number of months to include in the training set:",
    min_value=24,
    max_value=len(df),
    value=int(len(df) * 0.8),
    step=1
)
 
# Split the data
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

# 2) Decomposition Method Selection
st.subheader("2️⃣ Choose Decomposition Method")
decomposition_method = st.selectbox(
    "Select a decomposition method (Smoothing = seasonal_decompose OR STL; Autogregressive Lags = SARIMA; Dummy = Dummy Variable Regression):",
    options=["seasonal_decompose", "STL", "Dummy Variable Regression", "SARIMA"]
)

Model="Additive"

# Perform decomposition on the training data
if decomposition_method == "seasonal_decompose":
    Model = st.selectbox("Select seasonality model:", options=["Additive", "Multiplicative"])
    with st.spinner("Performing seasonal decomposition..."):
        result = seasonal_decompose(train_df["Composite"], model=Model.lower(), period=12)
        estimated_trend = result.trend
        estimated_seasonal = result.seasonal
        estimated_residual = result.resid
elif decomposition_method == "STL":
    with st.spinner("Performing STL decomposition..."):
        stl = STL(train_df["Composite"], period=12)
        result = stl.fit()
        estimated_trend = result.trend
        estimated_seasonal = result.seasonal
        estimated_residual = result.resid
elif decomposition_method == "Dummy Variable Regression":
    # Create dummy variables for each month
    with st.spinner("Performing Dummy Variable Decomposition..."):        
        # Prepare the data
        df_dummies = train_df.copy()
        df_dummies['Month'] = pd.DatetimeIndex(df_dummies.index).month
        df_dummies['Time'] = np.arange(len(df_dummies))  # Linear time trend
        # Create dummy variables for months
        month_dummies = pd.get_dummies(df_dummies['Month'], prefix='month', drop_first=True, dtype=float)
        # Combine trend and seasonal dummies
        X = pd.concat([df_dummies['Time'], month_dummies], axis=1)
        X = sm.add_constant(X)  # Adds intercept term
        y = df_dummies['Composite']
        # Fit the OLS model
        model = sm.OLS(y, X).fit()
        # Extract estimated components
        estimated_trend = model.params['Time'] * df_dummies['Time'] + model.params['const']
        estimated_seasonal = model.predict(X) - estimated_trend
        estimated_residual = y - model.predict(X)

elif decomposition_method == "SARIMA":
    # Fit SARIMA model
    st.sidebar.subheader("SARIMA Order")
    P = st.sidebar.slider("P (AR Component)", min_value=1, max_value=3, value=1, step=1)
    D = st.sidebar.slider("D (Differencing Component)", min_value=1, max_value=2, value=1, step=1)
    Q = st.sidebar.slider("Q (Moving Average Component)", min_value=1, max_value=4, value=1, step=1)
    with st.spinner("Performing SARIMA Decomposition..."):
        model = SARIMAX(train_df["Composite"], order=(1, 1, 1), seasonal_order=(P, D, Q, 12))
        result = model.fit(disp=False)
        #estimated_trend = result.trend if hasattr(result, 'trend') else pd.Series([0]*len(train_df), index=train_df.index)
        estimated_trend = pd.Series([0]*len(train_df), index=train_df.index)
        #estimated_seasonal = result.seasonal if hasattr(result, 'seasonal') else pd.Series([0]*len(train_df), index=train_df.index)
        estimated_seasonal = pd.Series([0]*len(train_df), index=train_df.index)
        estimated_residual = result.resid
        #estimated_seasonal = train_df["Composite"]-estimated_trend-estimated_residual

# 3) Component Comparison Charts
st.subheader("3️⃣ Compare True vs. Estimated Components")

# Align true and estimated components
aligned_index = estimated_trend.dropna().index
true_trend = train_df.loc[aligned_index, "Trend"]
true_seasonal = train_df.loc[aligned_index, "Seasonality"]
true_noise = train_df.loc[aligned_index, "Noise"]
true_trend_cycle=train_df.loc[aligned_index, "Trend_cycle"]
test=true_trend_cycle*(estimated_seasonal-1)
test_resid=true_trend_cycle*(estimated_residual-1)

# Create DataFrames for comparison
comparison_df = pd.DataFrame({
    "True Trend": true_trend,
    "Estimated Trend": estimated_trend.dropna(),
    "True Seasonality": true_seasonal,
    "Estimated Seasonality": estimated_seasonal.dropna(),
    "True Noise": true_noise,
    "Estimated Residual": estimated_residual.dropna(),
    "True Trend w Cycle": true_trend_cycle,
    "Estimated Seasonality Multiplicative": test,
    "Estimated Residual Multiplicative": test_resid
})
 
# Plot comparisons
st.write("**Trend Comparison**")
st.line_chart(comparison_df[["True Trend", "Estimated Trend","True Trend w Cycle"]])

st.write("**Seasonality Comparison**")
if Model=="Additive":
    st.line_chart(comparison_df[["True Seasonality", "Estimated Seasonality"]])
else:
    st.line_chart(comparison_df[["True Seasonality", "Estimated Seasonality Multiplicative"]])

st.write("**Noise/Residual Comparison**")
if Model=="Additive":
    st.line_chart(comparison_df[["True Noise", "Estimated Residual"]])
else:
    st.line_chart(comparison_df[["True Noise", "Estimated Residual Multiplicative"]])

st.header(" Forecasting on Test Set")

# Forecasting based on the selected decomposition method
with st.spinner("Generating forecasts..."):
    forecast_steps = len(test_df)
    forecast_index = test_df.index

    if decomposition_method == "seasonal_decompose":
        # Use last observed trend value
        last_trend=test_df["Trend_cycle"]
        # Repeat the last seasonal cycle
        seasonal_pattern = estimated_seasonal.dropna()[-12:]
        seasonal_forecast = np.tile(seasonal_pattern.values, int(np.ceil(forecast_steps / 12)))[:forecast_steps]
        # Forecast is sum of last trend and seasonal pattern
        if Model=="Additive":
            forecast_values = last_trend + seasonal_forecast
        else:
            forecast_values=last_trend*seasonal_forecast
        forecast_series = pd.Series(forecast_values, index=forecast_index)

    elif decomposition_method == "STL":
        from statsmodels.tsa.forecasting.stl import STLForecast
        stl_forecast = STLForecast(train_df["Composite"], model=sm.tsa.ARIMA, model_kwargs={"order": (1, 1, 1)}, period=12)
        stl_result = stl_forecast.fit()
        forecast_series = stl_result.forecast(steps=forecast_steps)
        forecast_series.index = forecast_index

    elif decomposition_method == "Dummy Variable Regression":
        with st.spinner("Generating forecasts using Dummy Variable Regression..."):
            # Prepare the training data
            df_train = train_df.copy()
            df_train['Month'] = pd.DatetimeIndex(df_train.index).month
            df_train['Time'] = np.arange(len(df_train))  # Linear time trend

            # Create dummy variables for months
            month_dummies_train = pd.get_dummies(df_train['Month'], prefix='month', drop_first=True, dtype=float)

            # Combine trend and seasonal dummies
            X_train = pd.concat([df_train['Time'], month_dummies_train], axis=1)
            X_train = sm.add_constant(X_train)  # Adds intercept term

            y_train = df_train['Composite']

            # Fit the OLS model
            model = sm.OLS(y_train, X_train).fit()

            # Prepare the test data
            df_test = test_df.copy()
            df_test['Month'] = pd.DatetimeIndex(df_test.index).month
            df_test['Time'] = np.arange(len(df_train), len(df_train) + len(df_test))  # Continue the time trend

            # Create dummy variables for months in test data
            month_dummies_test = pd.get_dummies(df_test['Month'], prefix='month', drop_first=True, dtype=float)

            # Ensure all dummy columns are present in test data
            for col in month_dummies_train.columns:
                if col not in month_dummies_test.columns:
                    month_dummies_test[col] = 0
            month_dummies_test = month_dummies_test[month_dummies_train.columns]  # Ensure same column order

            # Combine trend and seasonal dummies for test data
            X_test = pd.concat([df_test['Time'], month_dummies_test], axis=1)
            X_test = sm.add_constant(X_test)
            X_test = X_test[X_train.columns]  # Ensure same column order as training data

            # Generate forecasts
            forecast_values = model.predict(X_test)
            forecast_series = pd.Series(forecast_values.values, index=test_df.index)


    elif decomposition_method == "SARIMA":
        #sarima_model = SARIMAX(train_df["Composite"], order=(1, 1, 1), seasonal_order=(P, D, Q, 12))
        #sarima_result = sarima_model.fit(disp=False)
        #forecast_values = sarima_result.forecast(steps=forecast_steps)
        forecast_values=result.forecast(steps=forecast_steps)
        forecast_series = pd.Series(forecast_values.values, index=forecast_index)

# Combine actual and forecasted data for comparison
comparison_df = pd.DataFrame({
    "Actual": test_df["Composite"],
    "Forecast": forecast_series
})

# Combine training and test sets for the full composite series
full_series = pd.concat([train_df['Composite'], test_df['Composite']])

# Create a DataFrame for plotting
plot_df = pd.DataFrame({
    'Composite': full_series,
    'Forecast': forecast_series
})

# Plot the full composite series and forecasted values
st.subheader(" Forecast vs Actual with Historical Composite Series")
st.line_chart(plot_df)

# Plot the forecasts
st.subheader("🔮 Forecast vs Actual")
#st.line_chart(comparison_df)
#st.bar_chart(comparison_df,stack=False)


# Construct comparison DataFrame
comparison_df = pd.DataFrame({
    "Actual": test_df["Composite"],
    "Forecast": forecast_series
})
comparison_df["Date"] = comparison_df.index  # move index to column

# Reshape to long format
long_df = comparison_df.melt(id_vars="Date", var_name="Type", value_name="Value")

# Format Date for grouping (optional, simplifies x-axis)
long_df["DateStr"] = long_df["Date"].dt.strftime("%Y-%m")

# Create grouped bar chart
chart = alt.Chart(long_df).mark_bar().encode(
    x=alt.X('DateStr:N', title='Date'),
    xOffset=alt.X('Type:N'),
    y=alt.Y('Value:Q', title='Value'),
    color=alt.Color('Type:N', legend=alt.Legend(orient="bottom"))
).properties(
    width=700,
    height=400
)

# Display in Streamlit
st.altair_chart(chart, use_container_width=True)

# Calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    non_zero_indices = y_true != 0
    y_true = y_true[non_zero_indices]
    y_pred = y_pred[non_zero_indices]
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape_value = mean_absolute_percentage_error(test_df['Composite'], forecast_series)

# Display MAPE
st.subheader("Forecast Error")
st.metric(label="📉 Mean Absolute Percentage Error (MAPE)", value=f"{mape_value:.2f}%")
