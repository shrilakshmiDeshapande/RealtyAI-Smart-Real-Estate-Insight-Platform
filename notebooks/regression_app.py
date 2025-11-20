import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="RealtyAI - Price Prediction", layout="wide")
st.title("🏠 RealtyAI: Property Price Prediction")

@st.cache_resource
def load_model():
    return joblib.load(r"models/xgboost_price_model.pkl")

try:
    model = load_model()
    expected_features = model.feature_names_in_.tolist()
    st.success("✅ Model loaded!")
except Exception as e:
    st.error(f"❌ Model load failed: {e}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Upload CSV with property features", type=["csv"])

if uploaded_file:
    # Load and preview input
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Input Preview")
    st.write(f"Shape: {df.shape}")
    st.dataframe(df.head(10))  # Show first 10 rows
    
    # Predict button
    if st.button("🚀 Predict Prices"):
        try:
            has_actual = 'SalePrice' in df.columns
            
            # Validate features
            missing = set(expected_features) - set(df.columns)
            if missing:
                st.error(f"❌ Missing features: {sorted(missing)[:5]}...")
                st.stop()
            
            # Predict
            X = df[expected_features]
            preds = model.predict(X)
            
            # Build result with inputs + predictions
            result = pd.DataFrame({
                'GrLivArea': df['GrLivArea'],
                'OverallQual': df['OverallQual'],
                'GarageCars': df['GarageCars'],
                'FullBath': df['FullBath'],
                'YearBuilt': df['YearBuilt'],
                'LotArea': df['LotArea'],
                'Predicted_Price': preds
            })
            
            # Add actuals & errors if available
            if has_actual:
                actual = df['SalePrice']
                result['Actual_Price'] = actual
                result['Abs_Error'] = np.abs(actual - preds)
                result['Pct_Error'] = np.abs(actual - preds) / actual * 100
            
            # Show results
            st.subheader("✅ Prediction Results")
            st.dataframe(result)
            
            # Download button
            csv = result.to_csv(index=False)
            st.download_button(
                "📥 Download Predictions",
                csv,
                "realtyai_predictions.csv",
                "text/csv",
                key='download-csv'
            )
            
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")