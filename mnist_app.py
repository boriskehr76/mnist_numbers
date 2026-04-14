import streamlit as st
import numpy as np
import joblib
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Digit Classifier", page_icon="✦", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    background-color: #0a0a0a !important;
    color: #f0f0f0;
}

.stApp { background-color: #0a0a0a; }

.app-label {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.2em;
    color: #555;
    text-transform: uppercase;
    text-align: center;
    margin-bottom: 4px;
}

.app-title {
    font-family: 'Space Mono', monospace;
    font-size: 26px;
    font-weight: 700;
    color: #f0f0f0;
    text-align: center;
    margin-bottom: 2rem;
}

.result-digit {
    font-family: 'Space Mono', monospace;
    font-size: 72px;
    font-weight: 700;
    color: #e8ff47;
    text-align: center;
    line-height: 1;
}

.result-confidence {
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    color: #555;
    text-align: center;
    margin-top: 4px;
}

div[data-testid="stHorizontalBlock"] { gap: 10px; }

button[kind="secondary"] {
    background: transparent !important;
    border: 1px solid #2a2a2a !important;
    color: #888 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 0.1em !important;
    border-radius: 6px !important;
    width: 100% !important;
}

button[kind="secondary"]:hover {
    border-color: #444 !important;
    color: #bbb !important;
}

button[kind="primary"] {
    background: #e8ff47 !important;
    border: none !important;
    color: #0a0a0a !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    border-radius: 6px !important;
    width: 100% !important;
}

button[kind="primary"]:hover { background: #f0ff6a !important; }

/* Hide canvas toolbar */
.canvas-toolbar { display: none !important; }
</style>
""", unsafe_allow_html=True)

model = joblib.load('mnist_model.pkl')

st.markdown('<div class="app-label">neural network</div>', unsafe_allow_html=True)
st.markdown('<div class="app-title">Digit Classifier</div>', unsafe_allow_html=True)

col_left, col_right, col_right2 = st.columns([1, 3, 1])
with col_right:
    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = 0

    canvas = st_canvas(
        fill_color="black",
        stroke_width=22,
        stroke_color="white",
        background_color="black",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_key}"
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("NEW", type="secondary", use_container_width=True):
            st.session_state.canvas_key += 1
            st.session_state.prediction = ""
            st.session_state.confidence = ""
            st.rerun()

    with col2:
        if st.button("PREDICT →", type="primary", use_container_width=True):
            if canvas.image_data is not None:
                img = Image.fromarray(canvas.image_data.astype(np.uint8))
                img = img.convert('L')
                img = img.resize((28, 28))
                img_array = np.array(img).reshape(1, 784) / 255.0
                prediction = model.predict(img_array)
                probabilities = model.predict_proba(img_array)[0]
                confidence = probabilities.max() * 100
                st.session_state.prediction = str(prediction[0])
                st.session_state.confidence = f"{confidence:.1f}%"

    if st.session_state.get("prediction"):
        st.markdown(f'<div class="result-digit">{st.session_state.prediction}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-confidence">confidence: {st.session_state.confidence}</div>', unsafe_allow_html=True)