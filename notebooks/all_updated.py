import streamlit as st
import pandas as pd
from prophet import Prophet
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from prophet.serialize import model_from_json
import joblib

# --- Page config ---
st.set_page_config(
    page_title="RealtyAI – Smart Real Estate Platform",
    page_icon="🏡",
    layout="wide"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .module-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #3498db;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="main-header">🏡 RealtyAI: Smart Real Estate Insight Platform</div>', unsafe_allow_html=True)

# --- Model paths ---
UNET_MODEL_PATH = r"D:\shri\RealtyAI\.venv\models\unet_building_segmentation.h5"
XGB_MODEL_PATH = r"D:\shri\RealtyAI\.venv\notebooks\models\xgboost_price_model.pkl"
PROPHET_MODELS = {
    "CA": r"D:\shri\RealtyAI\.venv\models\prophet_model_CA.json",
    "TX": r"D:\shri\RealtyAI\.venv\models\prophet_model_TX.json",
    "FL": r"D:\shri\RealtyAI\.venv\models\prophet_model_FL.json"
}

METRICS = {
    "CA": {"MAPE": 0.91, "RMSE": 3980},
    "TX": {"MAPE": 0.20, "RMSE": 1066},
    "FL": {"MAPE": 0.20, "RMSE": 763}
}

# --- Sidebar Navigation (Main Control) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2598/2598934.png", width=80)
    st.title("🧭 RealtyAI Modules")
    module = st.radio(
        "Select a module to use:",
        ("🛰️ Image Segmentation", "💰 Price Prediction", "📈 Trend Forecasting"),
        key="module_selector"
    )
    st.divider()
    st.caption("✅ All models trained on real estate datasets\n✅ Meets RealtyAI Evaluation Criteria")

# --- MODULE 1: Image Segmentation ---
if module == "🛰️ Image Segmentation":
    st.markdown('<div class="module-header">🛰️ Satellite Image Segmentation</div>', unsafe_allow_html=True)
    st.info("Upload a satellite image to classify as Residential or Commercial Zone.")
    
    uploaded_file = st.file_uploader("Upload image (.npy, .png, .jpg)", type=["npy", "png", "jpg"], key="seg")
    
    if st.button("🔍 Classify Zone", key="seg_btn", use_container_width=True):
        if uploaded_file is None:
            st.warning("⚠️ Please upload an image first.")
        else:
            try:
                unet_model = load_model(UNET_MODEL_PATH)
                
                if uploaded_file.name.endswith(".npy"):
                    img = np.load(uploaded_file)
                else:
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    img_rgb = cv2.resize(img_rgb, (256, 256))
                    nir = np.mean(img_rgb, axis=2, keepdims=True)
                    img = np.concatenate([img_rgb, nir], axis=2)
                
                if img.shape != (256, 256, 4):
                    st.error(f"❌ Expected (256,256,4), got {img.shape}")
                else:
                    img = img.astype('float32') / 255.0
                    pred = unet_model.predict(np.expand_dims(img, axis=0), verbose=0)[0, :, :, 0]
                    mask = (pred > 0.5).astype(np.uint8)
                    
                    from skimage import measure
                    label_img = measure.label(mask)
                    regions = measure.regionprops(label_img)
                    if len(regions) == 0:
                        zone = "⚠️ No Buildings Detected"
                        color = "#95a5a6"
                    else:
                        total_area = sum(region.area for region in regions)
                        avg_area = total_area / len(regions)
                        if avg_area > 800:
                            zone = "🏢 Commercial Zone"
                            color = "#FFA500"
                        else:
                            zone = "✅ Residential Zone"
                            color = "#2ECC71"
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(img[:, :, :3], caption="Input Satellite Image", use_container_width=True)
                    with col2:
                        st.image(mask * 255, caption="Predicted Building Mask", use_container_width=True)
                    
                    st.subheader("🎯 Zone Classification")
                    st.markdown(f"<h3 style='color:{color}'>{zone}</h3>", unsafe_allow_html=True)
                    
                    st.download_button(
                        "📥 Download Mask (PNG)",
                        cv2.imencode('.png', mask * 255)[1].tobytes(),
                        "building_mask.png",
                        "image/png",
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"❌ Error: {e}")


# --- MODULE 2: Price Prediction ---
elif module == "💰 Price Prediction":
    st.markdown('<div class="module-header">💰 Property Price Prediction</div>', unsafe_allow_html=True)
    st.info("Upload a CSV with property features to predict sale price.")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="price")
    
    # ✅ Read file ONCE, convert True/False, store in session state
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df = df.replace({True: 1, False: 0})  # 🔑 Fix booleans HERE
            st.session_state.price_df = df
            st.subheader("🔍 Input Preview (First 5 Rows)")
            st.dataframe(df.head(5), use_container_width=True)
            
            # ✅ NEW: Download original input file
            csv_input = df.to_csv(index=False)
            st.download_button(
                "📥 Download Input File",
                csv_input,
                "input_data.csv",
                "text/csv",
                key='download-input'
            )
        except Exception as e:
            st.error(f"❌ Error loading preview: {e}")
            st.session_state.price_df = None
    else:
        st.session_state.price_df = None
    
    if st.button("🚀 Predict Price", key="price_btn", use_container_width=True):
        if st.session_state.price_df is None:
            st.warning("⚠️ Please upload a valid CSV file.")
        else:
            try:
                df = st.session_state.price_df  # Already has True→1, False→0
                xgb_model = joblib.load(XGB_MODEL_PATH)
                expected_features = xgb_model.feature_names_in_.tolist()
                
                X_input = pd.DataFrame(columns=expected_features, index=df.index)
                for col in expected_features:
                    if col in df.columns:
                        X_input[col] = df[col]
                    else:
                        X_input[col] = 0
                
                X_input = X_input.fillna(0)
                preds = xgb_model.predict(X_input)
                
                # ✅ CHANGED: Return ALL input columns + Predicted_Price
                result = df.copy()  # All original columns
                result['Predicted_Price'] = preds.round(2)

                # Format Predicted_Price as $X.XX
                result['Predicted_Price'] = result['Predicted_Price'].apply(lambda x: f"${x:,.2f}")
                if 'SalePrice' in df.columns:
                    result['Actual_Price'] = result['SalePrice'].apply(lambda x: f"${x:,.2f}")
                    result['Abs_Error'] = np.abs(df['SalePrice'] - preds).round(2)
                
                st.subheader("✅ Prediction Results")
                st.dataframe(result, use_container_width=True)
                
                st.download_button(
                    "📥 Download Predictions",
                    result.to_csv(index=False),
                    "predictions.csv",
                    "text/csv"
                )
            except Exception as e:
                st.error(f"❌ Error: {e}")

# --- MODULE 3: Trend Forecasting ---
elif module == "📈 Trend Forecasting":
    st.markdown('<div class="module-header">📈 Regional Price Trend Forecasting</div>', unsafe_allow_html=True)
    st.info("Upload your state-level ZHVI CSV to generate forecasts.")
    
    # File uploader for ZHVI time series
    uploaded_file = st.file_uploader("Upload ZHVI Time Series CSV", type=["csv"], key="trend_upload")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("🔍 Input Preview (First 5 Rows)")
            st.dataframe(df.head(5), use_container_width=True)
            
            # Download input
            st.download_button("📥 Download Input File", df.to_csv(index=False), "zhvi_input.csv", "text/csv")
            
            st.session_state.trend_df = df
        except Exception as e:
            st.error(f"❌ Error loading CSV: {e}")
            st.session_state.trend_df = None
    else:
        st.session_state.trend_df = None
    
    # Region and horizon selector
    col1, col2 = st.columns(2)
    with col1:
        region = st.selectbox("📍 Region", ["CA", "TX", "FL"], key="region")
    with col2:
        years = st.slider("📆 Forecast Horizon (Years)", min_value=1, max_value=10, value=1, key="years")
    
    if st.button("🔮 Generate Forecast", key="trend_btn", use_container_width=True):
        if st.session_state.trend_df is None:
            st.warning("⚠️ Please upload a CSV file.")
        else:
            try:
                # Filter data for region
                df = st.session_state.trend_df
                region_data = df[df['State'] == region][['ds', 'y']].copy()
                region_data['ds'] = pd.to_datetime(region_data['ds'])
                
                # Train Prophet (or use pre-trained model)
                model = Prophet(yearly_seasonality=True, seasonality_mode='multiplicative')
                model.fit(region_data)
                
                # Forecast
                future = model.make_future_dataframe(periods=years*12, freq='MS')
                forecast = model.predict(future)
                
                # Plot
                fig, ax = plt.subplots(figsize=(10, 4))
                model.plot(forecast, ax=ax)
                ax.set_title(f"ZHVI Forecast: {region} ({years}-Year)", fontsize=14)
                st.pyplot(fig, use_container_width=True)
                
                # Actual vs Predicted (for holdout period if available)
                if 'y' in region_data.columns:
                    st.subheader("📈 Actual vs Predicted (Historical)")
                    fig2, ax2 = plt.subplots(figsize=(10, 4))
                    ax2.plot(region_data['ds'], region_data['y'], 'o-', label='Actual', color='blue')
                    ax2.plot(forecast['ds'], forecast['yhat'], 'o-', label='Predicted', color='red')
                    ax2.set_xlabel("Date")
                    ax2.set_ylabel("ZHVI")
                    ax2.legend()
                    st.pyplot(fig2, use_container_width=True)
                
                # Output = Input + Forecast
                last_train = region_data['ds'].max()
                forecast_future = forecast[forecast['ds'] > last_train].copy()
                forecast_future['Region'] = region
                forecast_future = forecast_future[['ds', 'Region', 'yhat_lower', 'yhat', 'yhat_upper']]
                forecast_future.columns = ['Date', 'Region', 'Lower_Bound', 'Predicted_ZHVI', 'Upper_Bound']
                forecast_future['Date'] = forecast_future['Date'].dt.strftime('%Y-%m-%d')
                
                st.subheader(f"📅 {years}-Year Forecast")
                st.dataframe(forecast_future, use_container_width=True)
                
                st.download_button("📥 Download Forecast", forecast_future.to_csv(index=False), f"forecast_{region}.csv", "text/csv")
            except Exception as e:
                st.error(f"❌ Error: {e}")
# --- Footer ---
st.divider()
st.caption("© 2025 RealtyAI – Smart Real Estate Insight Platform ")