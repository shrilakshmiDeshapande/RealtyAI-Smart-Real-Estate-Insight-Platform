import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd

# --- Page config ---
st.set_page_config(page_title="RealtyAI - Image Segmentation", layout="wide")
st.title("🛰️ RealtyAI: Satellite Image Segmentation")
st.markdown("Upload a satellite image to detect buildings (residential/commercial zones).")

# --- Load UNet model (fixed naming conflict) ---
@st.cache_resource
def load_unet_model():
    return load_model(r"D:\shri\RealtyAI\.venv\models\unet_building_segmentation.h5")

try:
    model = load_unet_model()
    st.success("✅ UNet model loaded!")
except Exception as e:
    st.error(f"❌ Model load failed: {e}")
    st.stop()

# --- File uploader ---
uploaded_file = st.file_uploader("Upload satellite image (.npy, .png, or .jpg)", type=["npy", "png", "jpg"])

# --- Predict button ---
if st.button("🔍 Detect Buildings"):
    if uploaded_file is None:
        st.warning("Please upload an image first.")
    else:
        try:
            # Load image based on extension
            if uploaded_file.name.endswith(".npy"):
                img = np.load(uploaded_file)
            else:
                # Read PNG/JPG and convert to RGB
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_rgb = cv2.resize(img_rgb, (256, 256))
                # Add dummy NIR channel (mean of RGB)
                nir = np.mean(img_rgb, axis=2, keepdims=True)
                img = np.concatenate([img_rgb, nir], axis=2)
            
            # Validate shape
            if img.shape != (256, 256, 4):
                st.error(f"Image shape is {img.shape}. Expected (256, 256, 4).")
                st.stop()
            
            # Normalize to [0, 1]
            img = img.astype('float32') / 255.0
            
            # Predict
            pred = model.predict(np.expand_dims(img, axis=0), verbose=0)[0, :, :, 0]
            binary_mask = (pred > 0.5).astype(np.uint8)
            
            # Plot
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].imshow(img[:, :, :3])
            axes[0].set_title("Input Satellite Image")
            axes[0].axis('off')
            
            axes[1].imshow(binary_mask, cmap='gray')
            axes[1].set_title("Predicted Building Mask")
            axes[1].axis('off')
            
            st.pyplot(fig)
            
            # Download buttons
            st.subheader("📥 Download Results")
            
            # PNG mask
            mask_png = cv2.imencode('.png', binary_mask * 255)[1].tobytes()
            st.download_button(
                "Download Mask (PNG)",
                mask_png,
                "building_mask.png",
                "image/png"
            )
            
            # CSV mask
            mask_csv = pd.DataFrame(binary_mask).to_csv(index=False)
            st.download_button(
                "Download Mask (CSV)",
                mask_csv,
                "building_mask.csv",
                "text/csv"
            )
            
        except Exception as e:
            st.error(f"❌ Error: {e}")