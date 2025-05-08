import streamlit as st
import statsmodels.api as sm
import pandas as pd
import numpy as np
import datetime

# Set page configuration
st.set_page_config(page_title="Time Series Seasonality Simulator", layout="wide")

st.title("📈 Time Series Seasonality Simulator")

# Sidebar for user inputs
st.sidebar.header("Configure Time Series Components")

# Trend selection
trend_type = st.sidebar.selectbox(
    "Select Trend Type",
    options=["Linear", "Exponential", "Quadratic"]
)

# Trend parameters
if trend_type == "Linear":
    slope = st.sidebar.slider("Slope", min_value=-5.0, max_value=5.0, value=0.5, step=0.1)
    intercept = st.sidebar.slider("Intercept", min_value=-50.0, max_value=50.0, value=0.0, step=1.0)
elif trend_type == "Exponential":
    base = st.sidebar.slider("Base", min_value=0.1, max_value=5.0, value=1.1, step=0.1)
    rate = st.sidebar.slider("Rate", min_value=0.01, max_value=1.0, value=0.05, step=0.01)
elif trend_type == "Quadratic":
    a = st.sidebar.slider("Coefficient a", min_value=-1.0, max_value=1.0, value=0.01, step=0.01)
    b = st.sidebar.slider("Coefficient b", min_value=-5.0, max_value=5.0, value=0.5, step=0.1)
    c = st.sidebar.slider("Coefficient c", min_value=-50.0, max_value=50.0, value=0.0, step=1.0)

# Cycle parameters
st.sidebar.subheader("Cycle Component")
cycle_amplitude = st.sidebar.slider("Cycle Amplitude", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
cycle_period = st.sidebar.slider("Cycle Period (months)", min_value=6, max_value=60, value=24, step=1)
cycle_phase = st.sidebar.slider("Cycle Phase (radians)", min_value=0.0, max_value=2*np.pi, value=0.0, step=0.1)

# Seasonality parameters
st.sidebar.subheader("Seasonality Component")
seasonality_amplitude = st.sidebar.slider("Seasonality Amplitude", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

# Noise parameters
st.sidebar.subheader("Noise Component")
noise_std = st.sidebar.slider("Noise Standard Deviation", min_value=0.0, max_value=5.0, value=0.5, step=0.1)

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
    trend = base * np.exp(rate * t)
elif trend_type == "Quadratic":
    trend = a * t**2 + b * t + c

# Cycle
cycle = cycle_amplitude * np.sin(2 * np.pi * t / cycle_period + cycle_phase)

# Seasonality
seasonality = seasonality_amplitude * np.sin(2 * np.pi * t / 12)

# Noise
np.random.seed(42)  # For reproducibility
noise = np.random.normal(0, noise_std, n)

# Trend + Noise
trend_noise = trend+noise

# Seasonality + Cycle
season_cycle = seasonality + cycle

#trend_cycle = trend + cycle

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
st.subheader("📊 Composite Time Series")
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

from statsmodels.tsa.seasonal import seasonal_decompose, STL
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
    "Select a decomposition method:",
    options=["seasonal_decompose", "STL"]
)

# Perform decomposition on the training data
if decomposition_method == "seasonal_decompose":
    with st.spinner("Performing seasonal decomposition..."):
        result = seasonal_decompose(train_df["Composite"], model='additive', period=12)
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

# 3) Component Comparison Charts
st.subheader("3️⃣ Compare True vs. Estimated Components")

# Align true and estimated components
aligned_index = estimated_trend.dropna().index
true_trend = train_df.loc[aligned_index, "Trend"]
true_seasonal = train_df.loc[aligned_index, "Seasonality"]
true_noise = train_df.loc[aligned_index, "Noise"]
true_trend_cycle=df.loc[aligned_index, "Trend_cycle"]

# Create DataFrames for comparison
comparison_df = pd.DataFrame({
    "True Trend": true_trend,
    "Estimated Trend": estimated_trend.dropna(),
    "True Seasonality": true_seasonal,
    "Estimated Seasonality": estimated_seasonal.dropna(),
    "True Noise": true_noise,
    "Estimated Residual": estimated_residual.dropna()
})

# Plot comparisons
st.write("**Trend Comparison**")
st.line_chart(comparison_df[["True Trend", "Estimated Trend","True Trend + True Cycle"]])

st.write("**Seasonality Comparison**")
st.line_chart(comparison_df[["True Seasonality", "Estimated Seasonality"]])

st.write("**Noise/Residual Comparison**")
st.line_chart(comparison_df[["True Noise", "Estimated Residual"]])
