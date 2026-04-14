import streamlit as st
import numpy as np
import joblib
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.markdown("""
    <style>
    button[title="Undo"] svg, 
    button[title="Redo"] svg,
    button[title="Delete"] svg {
        fill: white !important;
    }
    </style>
""", unsafe_allow_html=True)


model = joblib.load('mnist_model.pkl')

st.title("MNIST Digit Classifier")
st.write("Draw a digit below and the model will classify it")

canvas = st_canvas(
    fill_color="black",
    stroke_width=20,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

if canvas.image_data is not None:
    img = Image.fromarray(canvas.image_data.astype(np.uint8))
    img = img.convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img).reshape(1, 784) / 255.0
    
    prediction = model.predict(img_array)
    st.markdown(f"## Predicted digit: {prediction[0]}")