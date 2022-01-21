from numpy.core.fromnumeric import argmax
import streamlit as st
from PIL import Image
import numpy as np
st.title("Image Classification")
st.header("Chest X-ray classification")
st.text("Upload Image for image classification")
from img_classification import classification
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