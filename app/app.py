import streamlit as st
from PIL import Image
from inference import predict

st.set_page_config(page_title="Pneumonia Detection AI", layout="centered")

st.title("🫁 AI Pneumonia Detection System")
st.write("Upload a Chest X-ray image for analysis")

uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "png", "jpeg"])

CLASS_NAMES = ["Normal", "Bacterial Pneumonia", "Viral Pneumonia"]

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            pred, confidence = predict(image)

        st.success(f"Prediction: {CLASS_NAMES[pred]}")
        st.info(f"Confidence: {confidence:.2f}")

        if pred != 0:
            st.warning("⚠️ Pneumonia detected. Please consult a medical professional.")