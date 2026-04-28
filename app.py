import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # hides TensorFlow warnings

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json

# =========================
# LOAD MODEL + LABELS
# =========================

@st.cache_resource
def load_emotion_model():
    return tf.keras.models.load_model("emotion_model.h5", compile=False)

@st.cache_data
def load_class_names():
    with open("class_names.json") as f:
        return json.load(f)

model = load_emotion_model()
class_names = load_class_names()

# =========================
# UI
# =========================
st.set_page_config(page_title="Happy vs Sad Classifier")

st.title("😊 Happy vs Sad Classifier")
st.write("Upload an image and get prediction")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# =========================
# PREDICTION FUNCTION
# =========================
def predict_image(image):
    # Ensure image is RGB before resizing
    image = image.convert("RGB")
    image = image.resize((128, 128))
    img = np.array(image)

    # handle RGBA images (if any)
    if img.shape[-1] == 4:
        img = img[:, :, :3]

    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)

    label = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    return label, confidence

# =========================
# RUN APP
# =========================
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    label, confidence = predict_image(image)

    st.markdown(f"### Prediction: {label}")
    st.write(f"Confidence: {confidence:.2f}")
