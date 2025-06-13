import streamlit as st
import pytesseract
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="AI Document Analyzer", layout="centered")

st.title("🧾 AI + XAI Document Analyzer")
st.write("Upload a document to see OCR + Model + XAI")

# Upload file
uploaded_file = st.file_uploader("Upload document image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert to OpenCV format
    img_np = np.array(image)
    text = pytesseract.image_to_string(img_np)

    st.subheader("🔍 OCR Extracted Text")
    st.code(text)

    st.subheader("🧠 Model Prediction (Coming Soon)")
    st.info("This will show your LayoutLM or Donut model prediction.")

    st.subheader("📈 SHAP Explanation (Coming Soon)")
    st.info("SHAP token attributions will be shown here later.")
