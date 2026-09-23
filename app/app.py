import streamlit as st
from PIL import Image
import os
import sys

# Allow imports from the project root (src/).
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference.schemas import HierarchicalPrediction
from src.inference.hierarchical_pipeline import HierarchicalPneumoniaPipeline
from src.inference.settings import InferenceSettings, InferenceSettingsError

st.set_page_config(page_title="Pneumonia Detection AI", layout="centered")

st.title("🫁 AI Pneumonia Detection System")
st.write(
    "Upload a chest X-ray for **hierarchical** AI analysis: "
    "first Normal vs Pneumonia, then (if pneumonia is detected) "
    "Bacterial vs Viral subtype."
)

# Sidebar with design context.
with st.sidebar:
    st.header("How it works")
    st.write(
        "1. A binary classifier detects whether the X-ray shows **Pneumonia** or is **Normal**.\n\n"
        "2. If pneumonia is detected, a second classifier predicts the subtype: **Bacterial** or **Viral**.\n\n"
        "This two-stage design follows the research finding that binary pneumonia "
        "detection is much stronger than direct three-class classification."
    )
    st.write("---")
    st.caption(
        "This Streamlit demo is a local prototype. "
        "The production system uses a FastAPI backend + Next.js frontend "
        "with hosted Hugging Face models."
    )

uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            try:
                settings = InferenceSettings.from_env()
                pipeline = HierarchicalPneumoniaPipeline.from_settings(settings)
                prediction: HierarchicalPrediction = pipeline.predict(image)
                result = prediction.to_dict()

                primary = result["primary_prediction"]
                primary_conf = result["primary_confidence"]
                subtype = result.get("subtype_prediction", "N/A")
                subtype_conf = result.get("subtype_confidence")
                probs = result.get("probabilities", {})
                disclaimer = result.get("disclaimer", "")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Primary Result")
                    if primary == "Pneumonia":
                        st.error(f"{primary} (confidence: {primary_conf:.2f})")
                    else:
                        st.success(f"{primary} (confidence: {primary_conf:.2f})")

                with col2:
                    st.markdown("#### Subtype")
                    if primary == "Pneumonia":
                        if subtype_conf is not None:
                            st.info(f"{subtype} (confidence: {subtype_conf:.2f})")
                        else:
                            st.info(f"{subtype}")
                    else:
                        st.caption("Not applicable (no pneumonia detected)")

                st.markdown("#### Probabilities")
                for label, prob in probs.items():
                    st.write(f"{label}: {prob:.2f}")

                st.info(disclaimer)

            except InferenceSettingsError as e:
                st.error(f"Configuration error: {e}")
                st.info(
                    "Set the required environment variables "
                    "(HF_BINARY_MODEL_ID, HF_SUBTYPE_MODEL_ID, HF_TOKEN, etc.) "
                    "or run the FastAPI backend (`uvicorn backend.app:app`) "
                    "and use the Next.js frontend."
                )
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.info(
                    "If you are running this demo without the backend, "
                    "make sure the Hugging Face environment variables are configured."
                )
