import streamlit as st
import numpy as np
import joblib
from PIL import Image
from streamlit_drawable_canvas import st_canvas

model = joblib.load('mnist_model.pkl')

st.title("MNIST Digit Classifier")
st.write("Draw a digit below and click Predict")

if "key" not in st.session_state:
    st.session_state.key = 0

canvas = st_canvas(
    fill_color="black",
    stroke_width=20,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key=f"canvas_{st.session_state.key}"
)

col1, col2 = st.columns(2)

with col1:
    if st.button("Predict"):
        if canvas.image_data is not None:
            img = Image.fromarray(canvas.image_data.astype(np.uint8))
            img = img.convert('L')
            img = img.resize((28, 28))
            img_array = np.array(img).reshape(1, 784) / 255.0

            prediction = model.predict(img_array)
            probabilities = model.predict_proba(img_array)[0]
            confidence = probabilities.max() * 100

            st.session_state.prediction = f"{prediction[0]}"
            st.session_state.confidence = f"{confidence:.1f}%"

with col2:
    if st.button("New"):
        st.session_state.key += 1
        st.session_state.prediction = ""
        st.session_state.confidence = ""
        st.rerun()

if "prediction" in st.session_state and st.session_state.prediction:
    st.markdown(f"## Predicted digit: {st.session_state.prediction}")
    st.markdown(f"### Confidence: {st.session_state.confidence}")