import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .main-header {
            text-align: center;
            padding: 1rem;
            margin-bottom: 2rem;
        }
        .upload-section {
            background-color: #f8f9fa;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .prediction-box {
            padding: 1.5rem;
            border-radius: 10px;
            margin-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Load model and class indices
@st.cache_resource
def load_model():
    working_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
    return tf.keras.models.load_model(model_path)

@st.cache_data
def load_class_indices():
    working_dir = os.path.dirname(os.path.abspath(__file__))
    return json.load(open(f"{working_dir}/class_indices.json"))

model = load_model()
class_indices = load_class_indices()

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence = float(predictions[0][predicted_class_index] * 100)
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name, confidence

# Main App UI
st.markdown("<div class='main-header'>", unsafe_allow_html=True)
st.title("🌿 Plant Disease Classifier")
st.markdown("Upload a plant leaf image to identify potential diseases")
st.markdown("</div>", unsafe_allow_html=True)

# Instructions
with st.expander("ℹ️ How to use"):
    st.markdown("""
    1. Click the 'Browse files' button below to upload an image of a plant leaf
    2. Make sure the image is clear and well-lit
    3. The image should be in JPG, JPEG, or PNG format
    4. Click 'Analyze' to get the prediction
    """)

# Upload Section
st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
uploaded_image = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear image of a plant leaf"
)

if uploaded_image is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Preview")
        image = Image.open(uploaded_image)
        st.image(image, use_column_width=True)
    
    with col2:
        st.markdown("### Analysis")
        if st.button('Analyze', type='primary'):
            with st.spinner('Analyzing image...'):
                try:
                    prediction, confidence = predict_image_class(model, uploaded_image, class_indices)
                    
                    st.markdown("<div class='prediction-box' style='background-color: #e8f4ea;'>", unsafe_allow_html=True)
                    st.markdown("#### Results")
                    st.markdown(f"**Detected Condition:** {prediction}")
                    st.markdown(f"**Confidence:** {confidence:.2f}%")
                    
                    # Progress bar for confidence
                    st.progress(confidence/100)
                    
                    if confidence < 70:
                        st.warning("⚠️ Low confidence prediction. Consider uploading a clearer image.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error("Error processing the image. Please try again with a different image.")
                    st.exception(e)
else:
    st.info("👆 Upload an image to get started")

st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>Created with ❤️ by Bhumika & Shashank </small>
</div>
""", unsafe_allow_html=True)