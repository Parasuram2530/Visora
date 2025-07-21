import streamlit as st
from PIL import Image
from inference import predict
import pyttsx3

engine = pyttsx3.init()
engine.setProperty('rate', 150)

st.set_page_config(page_title="Smart Vision Tagger", page_icon="🧠")
st.title("Smart Vision Tagger")
st.markdown("Upload an image and get AI-Powered tag Prediction!")
speak = st.checkbox("Enable voice Output", value=True)
uploaded_file = st.file_uploader("Uplaod Image", type=['jpg', 'jpeg', 'png'])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    with st.spinner("Predicting....."):
        class_name = predict(image)
    if speak:
        engine.say(f"The predicted Class is {class_name}")
        engine.runAndWait()
    else:
        st.success(f"Predicted class: {class_name}")




