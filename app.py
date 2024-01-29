import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np

# Load the trained model
model = load_model('InceptionV3.h5')

# Define the class labels
class_labels = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']

# Function to preprocess the image
IMAGE_SIZE = (224, 224)

def load_and_preprocess_image(image_path):
    try:
        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img)
        img = tf.image.resize(img, size=IMAGE_SIZE)
        return img
    except Exception as e:
        st.error(f"Error loading or preprocessing image: {e}")
        return None


def disease_predict(image_path):
    image = load_and_preprocess_image(image_path)

    if image is not None:
        try:
            pred = model.predict(tf.expand_dims(image, axis=0))
            if max(pred[0]) >= 0.90:
               predicted_value = class_labels[pred.argmax()]
               display_prediction(predicted_value, image)
            else:
                st.error("Invalid Image")
        except Exception as e:
            st.error(f"Error predicting disease: {e}")

def display_prediction(predicted_value, image):
    st.image(image.numpy() / 255., caption=f"Predicted Disease: {predicted_value}", use_column_width=True)
    st.success(f"Disease: **{predicted_value }**")



def disease_app():
    st.title('Agricultural Disease Detector 🦠')
    uploaded_file = st.file_uploader("Upload an Image for Disease Analysis", type="jpg")
    if uploaded_file is None:
        uploaded_file =  st.camera_input("Capture a photo")
    if uploaded_file is not None:
        img_path = f"uploaded_image.jpg"
        with open(img_path, "wb") as f:
            f.write(uploaded_file.read())
        disease_predict(img_path)
