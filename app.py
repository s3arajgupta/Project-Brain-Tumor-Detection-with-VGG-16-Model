"""Streamlit Interactive Radiologist Dashboard for NeuroScan Brain Tumor AI."""

from __future__ import annotations

import sys
from pathlib import Path

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
from PIL import Image
import streamlit as st

from neuroscan.models import DiagnosticLabel
from neuroscan.predictor import NeuroPredictor
from neuroscan.preprocess import crop_brain_contour, load_image

# Page configuration
st.set_page_config(
    page_title="NeuroScan AI - Brain Tumor Diagnostic Suite",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for clinical dashboard look
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0e1117;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #0284c7;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_predictor():
    return NeuroPredictor()


predictor = get_predictor()

# Header
st.markdown("<div class='main-header'>🧠 NeuroScan: Deep Learning & Explainable AI (Grad-CAM)</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>Automated Morphological Skull Stripping, VGG-16 Transfer Learning, and Class Activation Heatmaps for Clinical MRI Interpretation</div>",
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.header("🔬 Input MRI Selection")

dataset_dir = Path("brain_tumor_dataset")
sample_options = ["Upload your own scan"]

sample_files = {}
if dataset_dir.exists():
    yes_files = list((dataset_dir / "yes").glob("*.jpg")) + list((dataset_dir / "yes").glob("*.JPG"))
    no_files = list((dataset_dir / "no").glob("*.jpg")) + list((dataset_dir / "no").glob("*.JPG"))

    for p in yes_files[:5]:
        name = f"Sample: Tumor Positive ({p.name})"
        sample_options.append(name)
        sample_files[name] = p

    for p in no_files[:5]:
        name = f"Sample: Healthy / Negative ({p.name})"
        sample_options.append(name)
        sample_files[name] = p

selected_source = st.sidebar.selectbox("Choose Scan Source", sample_options)

active_image = None
if selected_source == "Upload your own scan":
    uploaded_file = st.sidebar.file_uploader("Upload Brain MRI (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        active_image = Image.open(uploaded_file)
else:
    file_path = sample_files[selected_source]
    active_image = Image.open(file_path)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Diagnostic Controls")
threshold = st.sidebar.slider("Diagnostic Threshold", min_value=0.10, max_value=0.90, value=0.50, step=0.05)
alpha = st.sidebar.slider("Grad-CAM Overlay Opacity", min_value=0.1, max_value=0.9, value=0.45, step=0.05)

if active_image is not None:
    img_rgb = np.array(active_image.convert("RGB"))

    # Run inference
    with st.spinner("Processing MRI through morphological contour cropper and VGG-16 backbone..."):
        report = predictor.predict(img_rgb)

    is_tumor = report.raw_score >= threshold
    label = "Tumor Detected" if is_tumor else "Healthy / No Tumor"
    confidence = report.raw_score if is_tumor else (1.0 - report.raw_score)

    # Top Metric Banner
    col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)

    with col_metric1:
        if is_tumor:
            st.error(f"⚠️ **Diagnosis:** {label}")
        else:
            st.success(f"✅ **Diagnosis:** {label}")

    with col_metric2:
        st.metric("Model Confidence", f"{confidence:.1%}")

    with col_metric3:
        st.metric("Raw Tumor Score", f"{report.raw_score:.2%}")

    with col_metric4:
        st.metric("Skull Cropping", "Active" if report.bbox else "Standard")

    st.markdown("---")

    # Image Visualization Grid
    st.subheader("🖼️ Explainable AI (XAI) Diagnostic Workflow")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("**1. Original MRI Scan**")
        st.image(active_image, use_container_width=True)
        st.caption(f"Dimensions: {active_image.size[0]}x{active_image.size[1]} px")

    with c2:
        st.markdown("**2. Skull-Stripped Parenchyma**")
        cropped_rgb, bbox = crop_brain_contour(img_rgb)
        st.image(cropped_rgb, use_container_width=True)
        st.caption(f"Cropped ROI: {bbox.width}x{bbox.height} px")

    with c3:
        st.markdown("**3. Grad-CAM Activation Map**")
        _, heatmap_224 = predictor.classifier.predict_with_gradcam(
            (np.array(Image.fromarray(cropped_rgb).resize((224, 224))).astype(np.float32) / 255.0)
        )
        st.image(heatmap_224, clamp=True, use_container_width=True)
        st.caption("Layer: block5_conv3 gradients")

    with c4:
        st.markdown("**4. Fused Diagnostic Overlay**")
        from neuroscan.gradcam import overlay_heatmap_on_image
        overlay = overlay_heatmap_on_image(cropped_rgb, heatmap_224, alpha=alpha)
        st.image(overlay, use_container_width=True)
        st.caption("Heatmap fused onto tissue")

    # Clinical Analysis Section
    st.markdown("---")
    st.subheader("📋 Clinical Findings & Model Interpretability")
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown(f"""
        * **Diagnostic Impression:** {report.notes}
        * **Explainability Rationale:** The Grad-CAM heatmap computes gradients from the final convolutional block (`block5_conv3`) to visualize which spatial features influenced the decision. 
          * **Hot spots (Red/Orange):** High neural activation indicative of abnormal tissue hyper-intensity.
          * **Cold regions (Blue/Purple):** Normal brain parenchyma and background tissue.
        * **Contour Extraction:** Automated extreme-point contour stripping eliminated background black borders, ensuring feature extraction focused on brain tissue rather than peripheral artifacts.
        """)

    with col_right:
        st.info("""
        **⚠️ Disclaimer:**
        This software is intended for research, educational, and developer portfolio demonstration purposes. It is not an FDA-cleared diagnostic medical device.
        """)
else:
    st.info("👈 Please upload an MRI scan or choose a sample scan from the sidebar to begin analysis.")
