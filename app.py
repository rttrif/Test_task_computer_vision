"""
DEPLOYING IMAGE CLASSIFIER
"""
import streamlit as st
from PIL import Image, ImageOps


st.title("КЛАССИФИКАТОР ДЕТАЛЕЙ ПО ФОТОГРАФИЯМ")
st.header("Классификация размера детали по двум классам: «маленькая» или «большая» деталь на изображении")
st.text("Загрузите изображение «большой» или «маленькой» детали")

from img_classification import part_classification

uploaded_file = st.file_uploader("Выберите фотографию детали ...", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption='Загруженная деталь', use_column_width=True)
    st.write("")
    st.write("Классификация ...")
    label = part_classification(image, "inception_part_classifier.h5")
    if label == 0:
        st.write("«Большая деталь»")
    else:
        st.write("«Маленькая деталь»")