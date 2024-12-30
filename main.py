import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
from util import classify, set_background

set_background("./data/bg5.png")

st.title("Pneumonia classification")

st.header("Please upload an image of a chest X-ray")

file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

model = load_model('./model/pneumonia_classifier.h5')

with open('./model/labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()

if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # classify
    class_name, conf_score = classify(image, model, class_names)

    st.write("## {}".format(class_name))
    st.write("### score: {}".format(conf_score))