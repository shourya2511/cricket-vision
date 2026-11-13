
import os
import sys
import json
import random
import tempfile
import time

import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
from torchvision.models import efficientnet_v2_s

# ==================================================
# CONFIGURATION
# ==================================================

PROJECT_ROOT = "/content/drive/MyDrive/cricket_vision"
sys.path.insert(0, PROJECT_ROOT)

from dataset_utils import preprocess_video

CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT, "checkpoints",
    "efficientnet_lstm_complete.pth"
)

EVALUATION_DIR = os.path.join(PROJECT_ROOT, "evaluation")

CLASS_NAMES = [
    "Cover Drive", "Pull", "Straight Drive",
    "Sweep", "Square Cut"
]

# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Cricket Vision | AI Analytics",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# CUSTOM CSS  (UI only - design tokens + component styles)
# ==================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ---------- DESIGN TOKENS ---------- */
:root {
    /* surfaces */
    --bg:            #0b1220;
    --surface:       #111c2e;
    --card:          #142238;
    --card-raised:   #1a2c45;
    --border:        #34506e;
    --border-soft:   #2a405a;

    /* text */
    --text:          #f1f5f9;   /* primary   */
    --text-2:        #cbd5e1;   /* secondary */
    --text-3:        #aebdd1;   /* tertiary / captions (still AA+ on all surfaces) */

    /* accents */
    --teal:          #4de0c1;
    --teal-bright:   #6af0ce;
    --teal-deep:     #20b99b;
    --on-teal:       #04141c;

    /* type scale */
    --fs-xs:   13px;
    --fs-sm:   14px;
    --fs-md:   16px;
    --fs-lg:   22px;
    --fs-xl:   30px;
    --fs-hero: 35px;
}

/* ---------- BASE ---------- */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: var(--bg);
    color: var(--text);
}

header[data-testid="stHeader"] {
    background: transparent;
}

header[data-testid="stHeader"] * {
    color: var(--text-2);
}

.block-container {
    padding-top: 2.5rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* ---------- TYPOGRAPHY ---------- */
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
    color: var(--text);
    font-weight: 700;
    letter-spacing: -0.01em;
}

.stApp h3 {
    font-size: var(--fs-lg);
    margin-top: 0.4rem;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stText"] {
    color: var(--text-2);
    font-size: var(--fs-md);
    line-height: 1.65;
}

[data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] b {
    color: var(--text);
}

[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p,
[data-testid="stImageCaption"],
.stApp small {
    color: var(--text-3) !important;
    font-size: var(--fs-xs) !important;
    line-height: 1.55;
}

/* widget labels (radio group title, uploader label, etc.) */
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label {
    color: var(--text) !important;
    font-size: var(--fs-sm);
    font-weight: 600;
}

hr {
    border-color: var(--border-soft) !important;
}

/* ---------- SIDEBAR ---------- */
section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border-soft);
}

section[data-testid="stSidebar"] * {
    color: var(--text-2);
}

section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: var(--text-3) !important;
    font-size: var(--fs-sm) !important;
}

section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: var(--teal) !important;
    font-size: var(--fs-xs);
    font-weight: 700;
    letter-spacing: 0.08em;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 8px 10px;
    border-radius: 8px;
    transition: background 0.15s ease;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(77, 224, 193, 0.10);
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label p {
    color: var(--text) !important;
    font-size: 15px;
    font-weight: 500;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: rgba(77, 224, 193, 0.14);
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p {
    color: var(--teal) !important;
    font-weight: 700;
}

.side-brand {
    color: var(--teal);
    font-size: 24px;
    font-weight: 800;
    letter-spacing: -0.01em;
    margin: 0 0 4px 0;
}

.side-heading {
    color: var(--teal);
    font-weight: 700;
    font-size: var(--fs-xs);
    letter-spacing: 0.08em;
    margin: 0 0 6px 0;
}

.side-info {
    color: var(--text) !important;
    font-size: var(--fs-sm);
    line-height: 2;
}

.side-foot {
    color: var(--text-3) !important;
    font-size: var(--fs-xs);
}

/* sidebar collapse / expand control stays visible */
[data-testid="stSidebarCollapseButton"] *,
[data-testid="stExpandSidebarButton"] * {
    color: var(--text-2) !important;
}

/* ---------- HERO ---------- */
.hero {
    background: linear-gradient(120deg, #142b43, #123d48);
    padding: 30px;
    border-radius: 18px;
    border: 1px solid #2f6177;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: var(--fs-hero);
    font-weight: 800;
    color: #ffffff;
    margin: 0 0 8px 0;
    padding: 0;
}

.hero p {
    color: #dbe7f1 !important;
    font-size: var(--fs-md);
    margin: 0;
}

/* ---------- SECTION TITLES ---------- */
.section-title {
    font-size: var(--fs-lg);
    font-weight: 700;
    color: var(--text);
    margin-top: 22px;
    margin-bottom: 14px;
}

/* ---------- CARDS ---------- */
.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px;
    min-height: 115px;
}

.metric-label {
    color: var(--text-2);
    font-size: var(--fs-sm);
    font-weight: 600;
}

.metric-value {
    color: var(--teal);
    font-size: var(--fs-xl);
    font-weight: 800;
    margin-top: 8px;
}

.result-card {
    background: linear-gradient(135deg, #123c43, #163047);
    border: 1px solid #2f8a7f;
    padding: 25px;
    border-radius: 16px;
    margin-top: 15px;
}

.result-label {
    color: #d3ecea;
    font-size: var(--fs-xs);
    font-weight: 600;
    letter-spacing: 0.06em;
}

.result-value {
    color: var(--teal-bright);
    font-size: 32px;
    font-weight: 800;
    margin-top: 4px;
    word-break: break-word;
}

.commentary-card {
    background: #172a3c;
    border-left: 4px solid var(--teal);
    padding: 20px;
    border-radius: 10px;
    color: var(--text) !important;
    font-size: 17px;
    line-height: 1.7;
}

.info-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 12px;
    color: var(--text);
}

.info-card b {
    color: var(--text);
    font-size: var(--fs-md);
    font-weight: 700;
}

.info-card hr {
    margin: 14px 0;
}

.shot-card {
    text-align: center;
    padding: 18px 10px;
}

.shot-name {
    color: var(--teal);
    font-weight: 700;
    font-size: var(--fs-md);
}

.small-muted {
    color: var(--text-2);
    font-size: var(--fs-sm);
}

/* ---------- BUTTONS ---------- */
div.stButton > button,
[data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-primary"] {
    background: var(--teal-deep);
    color: var(--on-teal) !important;
    border: 1px solid var(--teal-deep);
    border-radius: 9px;
    font-weight: 700;
    padding: 10px 22px;
    transition: background 0.15s ease, border-color 0.15s ease;
}

div.stButton > button p,
div.stButton > button span,
[data-testid="stBaseButton-secondary"] p,
[data-testid="stBaseButton-primary"] p {
    color: inherit !important;
    font-weight: 700;
    font-size: var(--fs-md);
}

div.stButton > button:hover,
[data-testid="stBaseButton-secondary"]:hover,
[data-testid="stBaseButton-primary"]:hover {
    background: #48dfbf;
    border-color: #48dfbf;
    color: var(--on-teal) !important;
}

div.stButton > button:focus-visible,
[data-testid="stBaseButton-secondary"]:focus-visible,
[data-testid="stBaseButton-primary"]:focus-visible {
    outline: 2px solid #ffffff;
    outline-offset: 2px;
}

/* ---------- FILE UPLOADER ---------- */
[data-testid="stFileUploaderDropzone"] {
    background: var(--card);
    border: 2px dashed #3f8f86;
    border-radius: 14px;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--teal);
    background: var(--card-raised);
}

[data-testid="stFileUploaderDropzone"] *,
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small,
[data-testid="stFileUploaderDropzoneInstructions"] div {
    color: var(--text-2);
}

[data-testid="stFileUploaderDropzone"] svg {
    color: var(--teal);
    fill: var(--teal);
}

/* button inside uploader must keep dark-on-teal text */
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploaderDropzone"] button * {
    color: var(--on-teal) !important;
}

[data-testid="stFileUploaderFile"],
[data-testid="stFileUploaderFileName"],
[data-testid="stFileUploaderFile"] small {
    color: var(--text) !important;
}

/* ---------- METRICS ---------- */
[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--border);
    padding: 16px 18px;
    border-radius: 12px;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] *,
[data-testid="stMetricLabel"] p {
    color: var(--text-2) !important;
    font-size: var(--fs-sm) !important;
    font-weight: 600 !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] * {
    color: var(--teal) !important;
    font-size: 28px !important;
    font-weight: 800 !important;
}

[data-testid="stMetricDelta"],
[data-testid="stMetricDelta"] * {
    color: var(--text-2) !important;
}

/* ---------- ALERTS (info / success / warning / error) ---------- */
[data-testid="stAlert"] {
    background: #16293f;
    border: 1px solid var(--border);
    border-left: 4px solid var(--teal);
    border-radius: 10px;
}

[data-testid="stAlert"] *,
[data-testid="stAlert"] p {
    color: var(--text) !important;
    font-size: var(--fs-md);
}

[data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) {
    border-left-color: #34d399;
}
[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) {
    border-left-color: #fbbf24;
}
[data-testid="stAlert"]:has([data-testid="stAlertContentError"]) {
    border-left-color: #f87171;
}
[data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) {
    border-left-color: #38bdf8;
}

/* ---------- PROGRESS / SPINNER ---------- */
[data-testid="stProgress"] > div > div,
.stProgress > div > div {
    background-color: #26364d;
}

.stProgress > div > div > div > div,
[data-testid="stProgress"] > div > div > div {
    background-color: #35d6b3;
}

[data-testid="stProgress"] p {
    color: var(--text-2) !important;
}

[data-testid="stSpinner"],
[data-testid="stSpinner"] * {
    color: var(--text-2) !important;
}

/* ---------- TABLES, CHARTS, IMAGES, VIDEO ---------- */
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
}

[data-testid="stTable"] th {
    background: var(--card-raised);
    color: var(--text) !important;
}

[data-testid="stTable"] td {
    color: var(--text-2) !important;
}

[data-testid="stVegaLiteChart"],
[data-testid="stArrowVegaLiteChart"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px;
}

[data-testid="stImage"] img {
    border-radius: 12px;
    border: 1px solid var(--border);
}

[data-testid="stVideo"] video {
    border-radius: 12px;
    border: 1px solid var(--border);
}

/* ---------- RESPONSIVE ---------- */
@media (max-width: 992px) {
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap;
        row-gap: 0.75rem;
    }
    [data-testid="stColumn"] {
        min-width: calc(50% - 1rem) !important;
    }
}

@media (max-width: 640px) {
    .block-container {
        padding: 1.5rem 1rem 2.5rem 1rem;
    }
    [data-testid="stColumn"] {
        min-width: 100% !important;
    }
    .hero {
        padding: 20px;
    }
    .hero h1 {
        font-size: 26px;
    }
    .hero p {
        font-size: var(--fs-sm);
    }
    .section-title {
        font-size: 19px;
    }
    .metric-value,
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] * {
        font-size: 24px !important;
    }
    .result-value {
        font-size: 26px;
    }
    .commentary-card {
        font-size: var(--fs-md);
        padding: 16px;
    }
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# MODEL ARCHITECTURE
# ==================================================

class LSTMClassifier(nn.Module):

    def __init__(
        self,
        input_size=1280,
        hidden_size=128,
        num_classes=5
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        lstm_out, (hidden, cell) = self.lstm(x)

        final_output = lstm_out[:, -1, :]

        return self.fc(self.dropout(final_output))


# ==================================================
# LOAD TRAINED MODEL
# ==================================================

@st.cache_resource
def load_models():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False
    )

    backbone = efficientnet_v2_s(weights=None)
    backbone.classifier = nn.Identity()

    backbone.load_state_dict(
        checkpoint["backbone_state_dict"]
    )

    backbone = backbone.to(device)
    backbone.eval()

    classifier = LSTMClassifier(
        input_size=checkpoint["feature_size"],
        hidden_size=checkpoint["hidden_size"],
        num_classes=checkpoint["num_classes"]
    )

    classifier.load_state_dict(
        checkpoint["lstm_state_dict"]
    )

    classifier = classifier.to(device)
    classifier.eval()

    return backbone, classifier, device


# ==================================================
# PREDICTION
# ==================================================

def predict_shot(video_path, backbone, classifier, device):

    video_tensor = preprocess_video(video_path)

    video_tensor = video_tensor.unsqueeze(0).to(device)

    with torch.no_grad():

        frames = video_tensor.squeeze(0)

        features = backbone(frames)

        features = features.unsqueeze(0)

        logits = classifier(features)

        probabilities = torch.softmax(logits, dim=1)

        confidence, predicted_idx = torch.max(
            probabilities, dim=1
        )

    return (
        CLASS_NAMES[predicted_idx.item()],
        confidence.item() * 100
    )


# ==================================================
# COMMENTARY
# ==================================================

COMMENTARY_TEMPLATES = {

    "Cover Drive": [
        "Beautifully played! A classic cover drive through the off side.",
        "Elegant cover drive! The batter finds the gap in the field.",
        "Excellent timing! A graceful cover drive from the batter."
    ],

    "Pull": [
        "Powerful pull shot! The batter rocks back and plays it confidently.",
        "What a pull! The ball races towards the leg side.",
        "Strong connection! A well-executed pull shot."
    ],

    "Straight Drive": [
        "Magnificent straight drive! The ball travels straight down the ground.",
        "Pure timing! A beautiful straight drive from the batter.",
        "Textbook cricket! The batter drives straight past the bowler."
    ],

    "Sweep": [
        "A well-executed sweep shot! The batter uses the pace to find the leg side.",
        "Excellent sweep! The batter gets down low and connects cleanly.",
        "Nicely played! A controlled sweep shot."
    ],

    "Square Cut": [
        "Brilliant square cut! The ball races through the point region.",
        "Short and wide, and punished! A crisp square cut.",
        "Excellent placement! A beautiful square cut through the off side."
    ]
}


def generate_commentary(shot):
    return random.choice(COMMENTARY_TEMPLATES[shot])


# ==================================================
# EVALUATION DATA
# ==================================================

def load_json(filename):

    path = os.path.join(EVALUATION_DIR, filename)

    with open(path, "r") as f:
        return json.load(f)


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown(
        "<div class='side-brand'>🏏 CRICKET VISION</div>",
        unsafe_allow_html=True
    )

    st.caption("AI-powered cricket analytics")

    st.divider()

    page = st.radio(
        "MAIN MENU",
        [
            "Home",
            "Shot Prediction",
            "Automatic Commentary",
            "Model Comparison",
            "Evaluation Dashboard"
        ]
    )

    st.divider()

    st.markdown(
        "<p class='side-heading'>MODEL INFORMATION</p>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class='side-info'>
            EfficientNetV2-S + LSTM<br>
            5 Cricket Shot Classes<br>
            15 Frames per Video
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        "<p class='side-foot'>Cricket Vision | Academic Project</p>",
        unsafe_allow_html=True
    )


# ==================================================
# COMMON HEADER
# ==================================================

st.markdown("""
<div class="hero">
    <h1>Cricket Vision 🏏</h1>
    <p>
        Intelligent cricket shot classification and
        automatic commentary using deep learning.
    </p>
</div>
""", unsafe_allow_html=True)


# ==================================================
# HOME PAGE
# ==================================================

if page == "Home":

    st.markdown(
        '<div class="section-title">Project Overview</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    cards = [
        ("5", "Shot Categories"),
        ("15", "Frames per Video"),
        ("72.92%", "Test Accuracy")
    ]

    for col, (value, label) in zip(
        [col1, col2, col3], cards
    ):

        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Supported Cricket Shots</div>',
        unsafe_allow_html=True
    )

    shot_cols = st.columns(5)

    for col, shot in zip(shot_cols, CLASS_NAMES):

        with col:
            st.markdown(f"""
            <div class="info-card shot-card">
                <div class="shot-name">
                    {shot}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">How It Works</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-card">
        <b>01 — Video Input</b><br>
        <span class="small-muted">
        Upload a cricket video containing a batting shot.
        </span>
        <hr>
        <b>02 — Frame Extraction</b><br>
        <span class="small-muted">
        Extract 15 uniformly sampled frames.
        </span>
        <hr>
        <b>03 — Feature Extraction</b><br>
        <span class="small-muted">
        EfficientNetV2-S extracts visual features.
        </span>
        <hr>
        <b>04 — Temporal Classification</b><br>
        <span class="small-muted">
        LSTM analyses the sequence and predicts the shot.
        </span>
        <hr>
        <b>05 — Commentary</b><br>
        <span class="small-muted">
        Generate a cricket-style commentary sentence.
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "Use the sidebar to explore shot prediction, "
        "commentary and model evaluation."
    )


# ==================================================
# SHOT PREDICTION PAGE
# ==================================================

elif page == "Shot Prediction":

    st.subheader("Shot Prediction")
    st.write(
        "Upload a cricket video and let the trained "
        "deep learning model identify the batting shot."
    )

    uploaded_file = st.file_uploader(
        "Choose your cricket video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_file is not None:

        st.video(uploaded_file)

        st.caption(
            "Recommended: a short clip containing one "
            "clearly visible batting shot."
        )

        if st.button("Analyse Video", type="primary"):

            temp_path = None

            try:

                suffix = os.path.splitext(
                    uploaded_file.name
                )[1]

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as temp_file:

                    temp_file.write(
                        uploaded_file.getbuffer()
                    )

                    temp_path = temp_file.name

                with st.spinner(
                    "Extracting frames and analysing the shot..."
                ):

                    backbone, classifier, device = load_models()

                    start = time.time()

                    shot, confidence = predict_shot(
                        temp_path,
                        backbone,
                        classifier,
                        device
                    )

                    inference_time = time.time() - start

                    commentary = generate_commentary(shot)

                st.session_state["last_shot"] = shot
                st.session_state["last_confidence"] = confidence
                st.session_state["last_commentary"] = commentary
                st.session_state["last_inference_time"] = inference_time

                st.success("Video analysis completed!")

                st.markdown("""
                <div class="section-title">
                    Prediction Results
                </div>
                """, unsafe_allow_html=True)

                c1, c2 = st.columns(2)

                with c1:
                    st.markdown(f"""
                    <div class="result-card">
                        <div class="result-label">
                            PREDICTED SHOT
                        </div>
                        <div class="result-value">
                            {shot}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with c2:
                    st.markdown(f"""
                    <div class="result-card">
                        <div class="result-label">
                            MODEL CONFIDENCE
                        </div>
                        <div class="result-value">
                            {confidence:.2f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.progress(confidence / 100)

                st.metric(
                    "Inference Time",
                    f"{inference_time:.2f} seconds"
                )

                st.markdown(
                    '<div class="section-title">Generated Commentary</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="commentary-card">{commentary}</div>',
                    unsafe_allow_html=True
                )

            except Exception as e:

                st.error(f"Prediction failed: {e}")

            finally:

                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)


# ==================================================
# AUTOMATIC COMMENTARY PAGE
# ==================================================

elif page == "Automatic Commentary":

    st.subheader("Automatic Commentary")

    if "last_shot" in st.session_state:

        st.markdown(
            f"### {st.session_state['last_shot']}"
        )

        st.caption(
            f"Confidence: "
            f"{st.session_state['last_confidence']:.2f}%"
        )

        st.markdown(
            f'<div class="commentary-card">'
            f'{st.session_state["last_commentary"]}'
            f'</div>',
            unsafe_allow_html=True
        )

        if st.button("Generate New Commentary"):

            st.session_state["last_commentary"] = (
                generate_commentary(
                    st.session_state["last_shot"]
                )
            )

            st.rerun()

    else:

        st.info(
            "First go to Shot Prediction and analyse a video. "
            "Your generated commentary will appear here."
        )


# ==================================================
# MODEL COMPARISON PAGE
# ==================================================

elif page == "Model Comparison":

    st.subheader("Temporal Model Comparison")

    lstm = load_json("lstm_raw_video_results.json")
    gru = load_json("gru_results.json")
    bilstm = load_json("bilstm_results.json")
    lstm_metrics = load_json("lstm_metrics.json")

    comparison = pd.DataFrame({

        "Model": [
            "EfficientNetV2-S + LSTM",
            "EfficientNetV2-S + GRU",
            "EfficientNetV2-S + BiLSTM"
        ],

        "Validation Accuracy (%)": [
            lstm_metrics["best_validation_accuracy"] * 100,
            gru["best_val_accuracy"] * 100,
            bilstm["best_val_accuracy"] * 100
        ],

        "Test Accuracy (%)": [
            lstm["test_accuracy"] * 100,
            gru["test_accuracy"] * 100,
            bilstm["test_accuracy"] * 100
        ],

        "Macro F1 (%)": [
            lstm["macro_f1"] * 100,
            gru["classification_report"]["macro avg"]["f1-score"] * 100,
            bilstm["classification_report"]["macro avg"]["f1-score"] * 100
        ]

    })

    st.dataframe(
        comparison.round(2),
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        '<div class="section-title">Test Accuracy</div>',
        unsafe_allow_html=True
    )

    chart = comparison.set_index("Model")[
        ["Test Accuracy (%)"]
    ]

    st.bar_chart(chart)

    st.markdown(
        '<div class="section-title">Macro F1 Score</div>',
        unsafe_allow_html=True
    )

    f1_chart = comparison.set_index("Model")[
        ["Macro F1 (%)"]
    ]

    st.bar_chart(f1_chart)

    st.info(
        "LSTM achieved the highest test accuracy (72.92%) "
        "and macro F1 (72.01%). GRU achieved the highest "
        "validation accuracy (72.44%)."
    )


# ==================================================
# EVALUATION DASHBOARD
# ==================================================

elif page == "Evaluation Dashboard":

    st.subheader("Model Evaluation Dashboard")

    metrics = load_json("lstm_raw_video_results.json")

    st.caption(
        f"{metrics['model']} | "
        f"{metrics['total_test_videos']} test videos"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Accuracy",
        f"{metrics['test_accuracy'] * 100:.2f}%"
    )

    c2.metric(
        "Macro Precision",
        f"{metrics['macro_precision'] * 100:.2f}%"
    )

    c3.metric(
        "Macro Recall",
        f"{metrics['macro_recall'] * 100:.2f}%"
    )

    c4, c5 = st.columns(2)

    c4.metric(
        "Macro F1",
        f"{metrics['macro_f1'] * 100:.2f}%"
    )

    c5.metric(
        "Weighted F1",
        f"{metrics['weighted_f1'] * 100:.2f}%"
    )

    st.markdown(
        '<div class="section-title">Confusion Matrix</div>',
        unsafe_allow_html=True
    )

    confusion_path = os.path.join(
        EVALUATION_DIR,
        "lstm_raw_video_confusion_matrix.png"
    )

    if os.path.exists(confusion_path):

        st.image(
            confusion_path,
            caption="LSTM — Test Set Confusion Matrix",
            use_container_width=True
        )

    else:

        st.warning("Confusion matrix image not found.")

    st.caption(
        "Evaluation performed on the complete 816-video "
        "test set. Rows represent actual classes and "
        "columns represent predicted classes."
    )
