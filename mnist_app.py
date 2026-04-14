import streamlit as st
import numpy as np
import joblib
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Digit Classifier", page_icon="✦", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');

html, body, .stApp, [class*="css"] {
    background-color: #0a0a0a !important;
    color: #f0f0f0;
}

/* Hide canvas toolbar completely */
.canvas-toolbar, [data-testid="stCanvasToolbar"],
canvas + div { display: none !important; }

/* Center everything */
[data-testid="stVerticalBlock"] {
    align-items: center;
}

/* Buttons */
.stButton > button {
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 0.1em !important;
    border-radius: 6px !important;
    height: 48px !important;
    width: 100% !important;
    transition: all 0.15s !important;
}

.stButton:first-child > button {
    background: transparent !important;
    border: 1px solid #2a2a2a !important;
    color: #888 !important;
}

.stButton:first-child > button:hover {
    border-color: #444 !important;
    color: #ccc !important;
}

.stButton:last-child > button {
    background: #e8ff47 !important;
    border: none !important;
    color: #0a0a0a !important;
    font-weight: 700 !important;
}

.stButton:last-child > button:hover {
    background: #f0ff6a !important;
}
</style>
""", unsafe_allow_html=True)

model = joblib.load('mnist_model.pkl')

# Header
st.markdown("""
    <div style="text-align:center; margin-bottom: 2rem;">
        <div style="font-family:'Space Mono',monospace; font-size:11px; letter-spacing:0.2em; color:#555; text-transform:uppercase; margin-bottom:6px;">neural network</div>
        <div style="font-family:'Space Mono',monospace; font-size:26px; font-weight:700; color:#f0f0f0;">Digit Classifier</div>
    </div>
""", unsafe_allow_html=True)

# Canvas — centered using columns
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

_, col, _ = st.columns([1, 2.8, 1])
with col:
    canvas = st_canvas(
        fill_color="black",
        stroke_width=22,
        stroke_color="white",
        background_color="black",
        width=280,
        height=280,
        drawing_mode="freedraw",
        display_toolbar=False,
        key=f"canvas_{st.session_state.canvas_key}"
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("NEW"):
            st.session_state.canvas_key += 1
            st.session_state.prediction = ""
            st.session_state.confidence = ""
            st.rerun()
    with col2:
        if st.button("PREDICT →"):
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
        st.markdown(f"""
            <div style="text-align:center; margin-top:1.5rem;">
                <div style="font-family:'Space Mono',monospace; font-size:72px; font-weight:700; color:#e8ff47; line-height:1;">{st.session_state.prediction}</div>
                <div style="font-family:'Space Mono',monospace; font-size:12px; color:#555; margin-top:6px;">confidence: {st.session_state.confidence}</div>
            </div>
        """, unsafe_allow_html=True)