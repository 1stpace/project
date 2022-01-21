from numpy.core.fromnumeric import argmax
import streamlit as st
from PIL import Image
import numpy as np
import keras
from PIL import Image, ImageOps
import numpy as np

st.title("Image Classification")
st.header("Chest X-ray classification")
st.text("Upload Image for image classification")
def classification(img,loadmodel):
    loadmodel = 'model.best.hdf5'
    model =  keras.models.load_model(loadmodel)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    return np.argmax(prediction)
classification()
uploaded_file = st.file_uploader("Choose an image ...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = classification(image, 'model.best.hdf5')
    if np.argmax(label) == 0:
        st.write("covid")
    else:
        st.write("others")
