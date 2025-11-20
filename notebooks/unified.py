import streamlit as st
import numpy as np
import cv2
import pandas as pd
from tensorflow.keras.models import load_model
from prophet.serialize import model_from_json
import joblib
import matplotlib.pyplot as plt

# --- Page config ---
st.set_page_config(page_title="RealtyAI Unified Dashboard", layout="wide")
st.title("🏡 RealtyAI: Unified Property Insights")

# --- Model paths ---
UNET_PATH = r"D:\shri\RealtyAI\.venv\models\unet_building_segmentation.h5"
PROPHET_MODELS = {
    "CA": r"D:\shri\RealtyAI\.venv\models\prophet_model_CA.json",
    "TX": r"D:\shri\RealtyAI\.venv\models\prophet_model_TX.json",
    "FL": r"D:\shri\RealtyAI\.venv\models\prophet_model_FL.json"
}

# --- Simple Price Calculator ---
def calculate_property_price(inputs, zone_type):
    """Simple but realistic property price calculation"""
    
    if zone_type == "Residential":
        # Base calculation for residential properties
        base_price = inputs.get('GrLivArea', 1500) * 125  # $125 per sq ft
        
        # Quality adjustments
        quality_multiplier = 1.0 + (inputs.get('OverallQual', 6) - 5) * 0.1
        base_price *= quality_multiplier
        
        # Feature bonuses
        feature_bonus = (
            inputs.get('GarageCars', 2) * 7500 +
            inputs.get('FullBath', 2) * 5000 +
            inputs.get('BedroomAbvGr', 3) * 3000 +
            inputs.get('TotRmsAbvGrd', 6) * 2000 +
            inputs.get('Fireplaces', 1) * 2500
        )
        
        # Lot value
        lot_value = inputs.get('LotArea', 10000) * 1.5
        
        total_price = base_price + feature_bonus + lot_value
        
    else:  # Commercial
        # Base calculation for commercial properties
        base_price = inputs.get('GrLivArea', 5000) * 100  # $100 per sq ft
        
        # Commercial features
        feature_bonus = (
            inputs.get('TotalBsmtSF', 2000) * 50 +
            inputs.get('1stFlrSF', 3000) * 60 +
            inputs.get('GarageArea', 1000) * 40 +
            inputs.get('Fireplaces', 1) * 5000
        )
        
        # Lot value (commercial lots are more valuable)
        lot_value = inputs.get('LotArea', 25000) * 3
        
        total_price = base_price + feature_bonus + lot_value
    
    # Age adjustment (newer = more expensive)
    current_year = 2024
    age = current_year - inputs.get('YearBuilt', 2000)
    age_discount = max(0, age * 0.005)  # 0.5% discount per year
    total_price *= (1 - age_discount)
    
    # Ensure reasonable minimum
    return max(total_price, 75000)

# --- Step 1: Upload Satellite Image ---
st.header("🛰️ Step 1: Upload Satellite Image")
uploaded_image = st.file_uploader("Upload satellite image (.npy, .png, .jpg)", type=["npy", "png", "jpg"])

zone_type = None
if uploaded_image:
    try:
        # Load and preprocess image
        if uploaded_image.name.endswith(".npy"):
            img = np.load(uploaded_image)
        else:
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_rgb = cv2.resize(img_rgb, (256, 256))
            nir = np.mean(img_rgb, axis=2, keepdims=True)
            img = np.concatenate([img_rgb, nir], axis=2)
        
        if img.shape != (256, 256, 4):
            st.error("❌ Expected (256,256,4) image")
        else:
            img = img.astype('float32') / 255.0
            unet_model = load_model(UNET_PATH)
            pred = unet_model.predict(np.expand_dims(img, axis=0), verbose=0)[0, :, :, 0]
            mask = (pred > 0.5).astype(np.uint8)
            
            # Classify zone
            from skimage import measure
            label_img = measure.label(mask)
            regions = measure.regionprops(label_img)
            if len(regions) == 0:
                zone_type = "No Buildings"
            else:
                avg_area = sum(r.area for r in regions) / len(regions)
                zone_type = "Commercial" if avg_area > 800 else "Residential"
            
            st.success(f"✅ Detected Zone: **{zone_type}**")
            col1, col2 = st.columns(2)
            with col1:
                st.image(img[:, :, :3], caption="Input Image", use_container_width=True)
            with col2:
                st.image(mask * 255, caption="Building Mask", use_container_width=True)
    except Exception as e:
        st.error(f"❌ Error: {e}")

# --- Step 2: Property Input Form ---
if zone_type and zone_type != "No Buildings":
    st.header("💰 Step 2: Enter Property Details")
    
    # Simple form with only essential features
    st.subheader("🏠 Essential Property Features")
    
    user_inputs = {}
    
    if zone_type == "Residential":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            user_inputs['GrLivArea'] = st.number_input(
                "Living Area (sq ft)", 
                min_value=500, 
                max_value=10000, 
                value=1500,
                help="Total above ground living area"
            )
            user_inputs['OverallQual'] = st.slider(
                "Overall Quality (1-10)",
                min_value=1,
                max_value=10,
                value=6,
                help="1=Very Poor, 10=Very Excellent"
            )
            user_inputs['BedroomAbvGr'] = st.number_input(
                "Bedrooms",
                min_value=0,
                max_value=8,
                value=3
            )
            
        with col2:
            user_inputs['FullBath'] = st.number_input(
                "Full Bathrooms",
                min_value=0,
                max_value=6,
                value=2
            )
            user_inputs['GarageCars'] = st.number_input(
                "Garage Cars",
                min_value=0,
                max_value=6,
                value=2
            )
            user_inputs['TotRmsAbvGrd'] = st.number_input(
                "Total Rooms",
                min_value=2,
                max_value=15,
                value=6
            )
            
        with col3:
            user_inputs['YearBuilt'] = st.number_input(
                "Year Built",
                min_value=1800,
                max_value=2024,
                value=2000
            )
            user_inputs['LotArea'] = st.number_input(
                "Lot Area (sq ft)",
                min_value=1000,
                max_value=100000,
                value=10000
            )
            user_inputs['Fireplaces'] = st.number_input(
                "Fireplaces",
                min_value=0,
                max_value=5,
                value=1
            )
            
    else:  # Commercial
        col1, col2, col3 = st.columns(3)
        
        with col1:
            user_inputs['GrLivArea'] = st.number_input(
                "Total Building Area (sq ft)",
                min_value=2000,
                max_value=50000,
                value=10000
            )
            user_inputs['LotArea'] = st.number_input(
                "Lot Area (sq ft)",
                min_value=5000,
                max_value=200000,
                value=25000
            )
            
        with col2:
            user_inputs['TotalBsmtSF'] = st.number_input(
                "Basement Area (sq ft)",
                min_value=0,
                max_value=20000,
                value=5000
            )
            user_inputs['1stFlrSF'] = st.number_input(
                "First Floor Area (sq ft)",
                min_value=1000,
                max_value=30000,
                value=8000
            )
            
        with col3:
            user_inputs['YearBuilt'] = st.number_input(
                "Year Built",
                min_value=1800,
                max_value=2024,
                value=2000
            )
            user_inputs['GarageArea'] = st.number_input(
                "Parking/Garage Area (sq ft)",
                min_value=0,
                max_value=10000,
                value=2000
            )
    
    if st.button("💰 Calculate Property Price", use_container_width=True):
        current_price = calculate_property_price(user_inputs, zone_type)
        st.session_state.current_price = current_price
        
        st.success(f"✅ Estimated Property Value: **${current_price:,.0f}**")
        
        # Show breakdown
        with st.expander("📊 See Calculation Details"):
            st.write(f"**Property Type:** {zone_type}")
            st.write(f"**Key Factors Considered:**")
            for key, value in user_inputs.items():
                st.write(f"- {key}: {value}")
            st.write(f"**Final Estimate:** ${current_price:,.0f}")

# --- Step 3: Forecast Horizon ---
if 'current_price' in st.session_state:
    st.header("📈 Step 3: Forecast Future Price")
    col1, col2 = st.columns(2)
    with col1:
        region = st.selectbox("📍 Region", ["CA", "TX", "FL"])
    with col2:
        years = st.slider("📆 Forecast Years", 1, 10, 1)
    
    if st.button("🔮 Generate Forecast", use_container_width=True):
        try:
            with open(PROPHET_MODELS[region], "r") as f:
                prophet_model = model_from_json(f.read())
            
            periods = years * 12
            future = prophet_model.make_future_dataframe(periods=periods, freq='MS')
            forecast = prophet_model.predict(future)
            
            # Get last forecasted value
            future_dates = forecast[forecast['ds'] > pd.Timestamp.now()]
            if len(future_dates) > 0:
                future_price = future_dates['yhat'].iloc[-1]
            else:
                future_price = forecast['yhat'].iloc[-1]
            
            # Apply forecast to current price
            current_base = st.session_state.current_price
            growth_factor = future_price / 200000  # Normalize to typical home price
            adjusted_future_price = current_base * growth_factor
            
            growth = (adjusted_future_price - current_base) / current_base * 100
            
            st.subheader("📊 Forecast Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Value", f"${current_base:,.0f}")
            with col2:
                st.metric(f"Future Value ({years} year)", f"${adjusted_future_price:,.0f}")
            with col3:
                st.metric("Projected Growth", f"{growth:.1f}%")
            
            # Simple plot
            fig, ax = plt.subplots(figsize=(10, 4))
            
            # Create a simple projection line
            years_range = list(range(0, years + 1))
            prices = [current_base * (1 + growth/100 * (y/years)) for y in years_range]
            
            ax.plot(years_range, prices, 'b-', linewidth=2, marker='o')
            ax.fill_between(years_range, prices, alpha=0.2)
            ax.set_xlabel('Years')
            ax.set_ylabel('Property Value ($)')
            ax.set_title(f'Property Value Projection: {region} Region')
            ax.grid(True, alpha=0.3)
            
            # Format y-axis as currency
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            st.pyplot(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error in forecast: {e}")
            # Fallback simple growth calculation
            st.info("📈 Using standard market growth rates")
            
            # Typical annual growth rates by region
            growth_rates = {"CA": 0.05, "TX": 0.04, "FL": 0.045}
            annual_growth = growth_rates.get(region, 0.04)
            
            future_price = st.session_state.current_price * (1 + annual_growth) ** years
            growth = (future_price - st.session_state.current_price) / st.session_state.current_price * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Value", f"${st.session_state.current_price:,.0f}")
            with col2:
                st.metric(f"Future Value ({years} year)", f"${future_price:,.0f}")
            with col3:
                st.metric("Projected Growth", f"{growth:.1f}%")