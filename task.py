import streamlit as st
import numpy as np
import cv2

def analyse_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray

st.title("Image Analyser")

uploaded_image = st.file_uploader("Upload an image", type=["jpeg", "jpg", "png"], accept_multiple_files=False)

if uploaded_image is not None:
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    if st.button("Analyse Image"):
        # st.write("Analysing image...")
        analysed_image = analyse_image(image)
        st.image(analysed_image, caption='Analysed Image.', use_column_width=True)
