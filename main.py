import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('weather_model.h5')
    return model

model = load_model()

# Set up the title of the web app
st.write("# Weather Classification System")

# File uploader widget
file = st.file_uploader("Choose a weather condition photo from your computer", type=["jpg", "png"])

# Function to import and predict the class of the image
def import_and_predict(image_data, model):
    size = (244, 244)  # Define the target size
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)  # Resize and resample the image
    img = np.asarray(image)  # Convert the image to a NumPy array
    img_reshape = img[np.newaxis,...]  # Add an extra dimension for batch size
    prediction = model.predict(img_reshape)  # Make a prediction
    return prediction

# Display instructions or the image and prediction
if file is None:
    st.text("Please upload an image file.")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_names = ['Rain', 'Shine', 'Cloudy', 'Sunrise']
    string = "This image most likely shows: " + class_names[np.argmax(prediction)]
    st.success(string)
