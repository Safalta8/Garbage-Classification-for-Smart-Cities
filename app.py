import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Garbage Classification for Smart Cities",
    page_icon="♻️",
    layout="wide"
)

# ==================================================
# CUSTOM CSS
# ==================================================
st.markdown("""
<style>

/* ===== MAIN APP ===== */
.stApp {
    background-color: #D8E4F0;
}

/* ===== FIX TOP SPACE ===== */
.block-container {
    padding-top: 2rem;
}

/* ===== TITLE ===== */
.main-title {
    text-align: center;
    font-size: 52px;
    font-weight: 900;
    color: #1C2B48;
    margin-top: 0px;
    margin-bottom: 8px;
    text-shadow: 2px 2px 8px rgba(255,255,255,0.7);
}

/* ===== SUBTITLE ===== */
.sub-title {
    text-align: center;
    font-size: 22px;
    font-weight: 500;
    color: #243B63;
    margin-bottom: 25px;
}

/* ===== SIDEBAR ===== */
[data-testid="stSidebar"] {
    background: linear-gradient(
        to bottom,
        #1C2B48,
        #4F6D99
    );
}

[data-testid="stSidebar"] * {
    color: white !important;
}

/* ===== FILE UPLOADER ===== */
[data-testid="stFileUploader"] {
    background: #D8E4F0;
    border: 2px dashed #7CA7E7;
    border-radius: 20px;
    padding: 12px;
}

/* ===== RESULT BOX ===== */
.result-box {
    background: linear-gradient(
        to right,
        #1C2B48,
        #4F6D99
    );
    color: white;
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    font-size: 30px;
    font-weight: bold;
}

/* ===== PROGRESS BAR ===== */
.stProgress > div > div > div > div {
    background-color: #1C2B48;
}

/* ===== INFO BOX ===== */
.stInfo {
    background-color: #A7C7E7 !important;
    color: #1C2B48 !important;
    border-radius: 12px;
}

/* ===== IMAGE ===== */
img {
    border-radius: 15px;
}

/* ===== REMOVE EMPTY WHITE BOXES ===== */
[data-testid="stHorizontalBlock"] > div:empty {
    display: none !important;
}

/* ===== FOOTER ===== */
.footer {
    text-align: center;
    color: #1C2B48;
    font-size: 18px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.title("♻️ Smart City")

st.sidebar.markdown("---")

st.sidebar.subheader("📌 Project Details")

st.sidebar.write("""
This AI system classifies garbage into:

♻️ **Biodegradable**

🚯 **Non-Biodegradable**

using **Deep Learning (MobileNetV2)**.
""")

st.sidebar.markdown("---")

st.sidebar.success("✅ Smart Waste Management")

# ==================================================
# LOAD MODEL
# ==================================================
MODEL_PATH = "best_model.keras"

@st.cache_resource
def load_my_model():
    try:
        return load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_my_model()

# ==================================================
# TITLE
# ==================================================
st.markdown("""
<h1 class="main-title">
♻️ Garbage Classification for Smart Cities
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<p class="sub-title">
Upload a waste image and classify it using AI
</p>
""", unsafe_allow_html=True)

# ==================================================
# FILE UPLOAD
# ==================================================
uploaded_file = st.file_uploader(
    "📤 Upload Waste Image",
    type=["jpg", "jpeg", "png"]
)

# ==================================================
# PREDICTION
# ==================================================
if uploaded_file is not None and model is not None:

    col1, col2 = st.columns(2)

    # LEFT COLUMN
    with col1:
        img = Image.open(uploaded_file)

        st.image(
            img,
            caption="Uploaded Waste Image",
            use_container_width=True
        )

    # IMAGE PREPROCESSING
    img = img.convert("RGB")
    img_resize = img.resize((224, 224))

    img_array = image.img_to_array(img_resize)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # MODEL PREDICTION
    try:
        prediction = model.predict(
            img_array,
            verbose=0
        )[0][0]

        # LABEL
        if prediction < 0.5:
            result = "♻️ Biodegradable"
            confidence = (1 - prediction) * 100
        else:
            result = "🚯 Non-Biodegradable"
            confidence = prediction * 100

        # RIGHT COLUMN
        with col2:

            st.subheader("🔍 Prediction Result")

            st.markdown(f"""
            <div class='result-box'>
                {result}
            </div>
            """, unsafe_allow_html=True)

            st.write("")

            st.progress(int(confidence))

            st.info(
                f"Confidence Score: {confidence:.2f}%"
            )

    except Exception as e:
        st.error(f"Prediction Error: {e}")

# ==================================================
# FOOTER
# ==================================================
st.write("---")

st.markdown("""
<div class="footer">
AI Powered Waste Classification ♻️
</div>
""", unsafe_allow_html=True)