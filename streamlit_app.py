import streamlit as st
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

# Composite time series
composite = trend + cycle + seasonality + noise

# Create DataFrame
df = pd.DataFrame({
    "Date": dates,
    "Composite": composite,
    "Trend": trend,
    "Cycle": cycle,
    "Seasonality": seasonality,
    "Noise": noise
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

    st.write("**Seasonality Component**")
    st.line_chart(df["Seasonality"])

with col2:
    st.write("**Cycle Component**")
    st.line_chart(df["Cycle"])

    st.write("**Noise Component**")
    st.line_chart(df["Noise"])
