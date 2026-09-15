import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import plotly.graph_objects as go

# ==========================================
# 1. Page Configuration & Custom CSS (UI/UX)
# ==========================================
st.set_page_config(
    page_title="Energy Demand Forecaster & XAI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimalistic CSS Styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid #4F46E5;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Data & Model Caching
# ==========================================
@st.cache_data
def load_data_and_model():
    # Synthetic energy consumption dataset generation
    dates = pd.date_range(start="2025-01-01", periods=1000, freq="h")
    np.random.seed(42)
    
    temp = 20 + 10 * np.sin(np.linspace(0, 10, 1000)) + np.random.normal(0, 2, 1000)
    humidity = 50 + 20 * np.cos(np.linspace(0, 10, 1000)) + np.random.normal(0, 5, 1000)
    is_weekend = (dates.dayofweek >= 5).astype(int)
    hour = dates.hour
    
    # Target variable: Energy Consumption (kWh)
    demand = (
        100 
        + 3.5 * temp 
        - 0.5 * humidity 
        + 15 * np.sin(2 * np.pi * hour / 24) 
        - 20 * is_weekend 
        + np.random.normal(0, 5, 1000)
    )
    
    df = pd.DataFrame({
        'temperature': temp,
        'humidity': humidity,
        'hour': hour,
        'is_weekend': is_weekend,
        'demand': demand
    }, index=dates)
    
    X = df[['temperature', 'humidity', 'hour', 'is_weekend']]
    y = df['demand']
    
    # Train XGBoost Model
    model = xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X, y)
    
    # SHAP Explainer
    explainer = shap.TreeExplainer(model)
    
    return df, X, model, explainer

df, X, model, explainer = load_data_and_model()

# ==========================================
# 3. Sidebar — Input Parameters
# ==========================================
st.sidebar.image("https://img.icons8.com/color/96/lightning-bolt.png", width=60)
st.sidebar.title("Forecast Parameters")
st.sidebar.caption("Adjust feature values to calculate energy demand")

input_temp = st.sidebar.slider("Temperature (°C)", min_value=-10.0, max_value=40.0, value=25.0, step=0.5)
input_humidity = st.sidebar.slider("Humidity (%)", min_value=10.0, max_value=100.0, value=50.0, step=1.0)
input_hour = st.sidebar.slider("Hour of Day", min_value=0, max_value=23, value=14)
input_is_weekend = st.sidebar.selectbox("Day Type", options=["Weekday", "Weekend"])

is_weekend_val = 1 if input_is_weekend == "Weekend" else 0

# Construct input vector for inference
user_data = pd.DataFrame({
    'temperature': [input_temp],
    'humidity': [input_humidity],
    'hour': [input_hour],
    'is_weekend': [is_weekend_val]
})

# ==========================================
# 4. Main UI Dashboard
# ==========================================
st.title("⚡ Energy Demand Forecaster & XAI")
st.markdown("Interactive energy demand forecasting platform powered by *Explainable AI*.")

st.divider()

# Compute prediction
prediction = model.predict(user_data)[0]

# Key Performance Indicators (KPIs)
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Predicted Demand", value=f"{prediction:.2f} kWh")
with col2:
    mean_demand = df['demand'].mean()
    delta = prediction - mean_demand
    st.metric(label="Deviation from Mean", value=f"{delta:+.2f} kWh", delta_color="normal")
with col3:
    st.metric(label="Grid Status", value="Optimal" if prediction < 180 else "High Load")

st.write("")

# Navigation Tabs
tab1, tab2, tab3 = st.tabs(["📊 Forecast & Time Series", "🧩 SHAP Interpretation (XAI)", "📁 Dataset Overview"])

# --- TAB 1: Time Series Plot ---
with tab1:
    st.subheader("Historical Series & Prediction Point")
    
    fig = go.Figure()
    # Historical data trace
    fig.add_trace(go.Scatter(
        x=df.index[-72:], 
        y=df['demand'][-72:], 
        mode='lines', 
        name='Historical Data (Last 3 Days)',
        line=dict(color='#4F46E5', width=2)
    ))
    # Current prediction point
    fig.add_trace(go.Scatter(
        x=[df.index[-1]], 
        y=[prediction], 
        mode='markers', 
        name='Current Prediction',
        marker=dict(color='#EF4444', size=12, symbol='star')
    ))
    
    fig.update_layout(
        template="plotly_white",
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: SHAP (Explainable AI) ---
with tab2:
    st.subheader("Model Decision Explanation")
    st.caption("Visualizing feature contributions towards the final prediction.")
    
    # Calculate SHAP values for current user input
    shap_values = explainer(user_data)
    
    # Prepare data for SHAP bar plot
    feature_names = ['Temperature', 'Humidity', 'Hour of Day', 'Weekend']
    contributions = shap_values.values[0]
    
    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'Contribution (SHAP value)': contributions
    }).sort_values(by='Contribution (SHAP value)', ascending=True)
    
    # Plotly horizontal bar chart
    colors = ['#EF4444' if x < 0 else '#10B981' for x in shap_df['Contribution (SHAP value)']]
    
    fig_shap = go.Figure(go.Bar(
        x=shap_df['Contribution (SHAP value)'],
        y=shap_df['Feature'],
        orientation='h',
        marker_color=colors
    ))
    
    fig_shap.update_layout(
        template="plotly_white",
        height=350,
        xaxis_title="Impact on Forecast (kWh)",
        margin=dict(l=20, r=20, t=30, b=20)
    )
    
    st.plotly_chart(fig_shap, use_container_width=True)
    
    st.info("""
    💡 *How to interpret this chart:* 
    * *Green bars (Right):* Feature increases predicted energy demand.
    * *Red bars (Left):* Feature decreases predicted energy demand.
    """)

# --- TAB 3: Dataset View ---
with tab3:
    st.subheader("Raw Dataset Preview")
    st.dataframe(df.tail(10), use_container_width=True)

# ==========================================
# 5. Footer
# ==========================================
st.divider()
st.caption("AI Engineer Portfolio Project | Developed with Streamlit, XGBoost & SHAP")