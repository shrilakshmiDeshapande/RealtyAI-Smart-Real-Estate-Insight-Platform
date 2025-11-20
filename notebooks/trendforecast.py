import streamlit as st
import pandas as pd
from prophet.serialize import model_from_json
import matplotlib.pyplot as plt

st.set_page_config(page_title="RealtyAI - Price Trend Forecasting", layout="wide")
st.title("📈 RealtyAI: Regional Price Trend Forecasting")
st.markdown("Select a region and click **Predict** to view price trend forecasts.")

# --- Model paths ---
MODEL_PATHS = {
    "CA": r"D:\shri\RealtyAI\.venv\models\prophet_model_CA.json",
    "TX": r"D:\shri\RealtyAI\.venv\models\prophet_model_TX.json",
    "FL": r"D:\shri\RealtyAI\.venv\models\prophet_model_FL.json"
}

# --- Precomputed metrics ---
METRICS = {
    "CA": {"MAPE": 0.91, "RMSE": 3980},
    "TX": {"MAPE": 0.20, "RMSE": 1066},
    "FL": {"MAPE": 0.20, "RMSE": 763}
}

# --- Region selector (no auto-action) ---
region = st.selectbox("Select Region", options=["CA", "TX", "FL"])

# --- Predict button ---
if st.button("🚀 Predict Trend"):
    if not region:
        st.warning("Please select a region.")
    else:
        try:
            # Load model
            with open(MODEL_PATHS[region], "r") as f:
                model = model_from_json(f.read())
            
            # Generate 2018 forecast
            future = pd.DataFrame({
                'ds': pd.date_range(start='2018-01-01', periods=12, freq='MS')
            })
            forecast = model.predict(future)
            
            # Plot
            fig, ax = plt.subplots(figsize=(12, 6))
            model.plot(forecast, ax=ax)
            ax.set_title(f"ZHVI Trend Forecast: {region} (2018)", fontsize=14)
            ax.set_xlabel("Year")
            ax.set_ylabel("ZHVI (Median Home Value)")
            st.pyplot(fig)
            
            # Metrics
            st.subheader("📊 Model Performance (2017 Holdout)")
            col1, col2 = st.columns(2)
            col1.metric("MAPE", f"{METRICS[region]['MAPE']:.2f}%")
            col2.metric("RMSE", f"${METRICS[region]['RMSE']:,.0f}")
            
            # Forecast table
            st.subheader("📅 12-Month Forecast (2018)")
            forecast_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
            forecast_table.columns = ['Date', 'Predicted_ZHVI', 'Lower_Bound', 'Upper_Bound']
            forecast_table['Predicted_ZHVI'] = forecast_table['Predicted_ZHVI'].round().astype(int)
            st.dataframe(forecast_table)
            
            # Download
            csv = forecast_table.to_csv(index=False)
            st.download_button(
                "📥 Download Forecast",
                csv,
                f"forecast_{region}.csv",
                "text/csv"
            )
            
        except Exception as e:
            st.error(f"❌ Error: {e}")