# Cricket Vision – Automatic Cricket Shot Classification

Cricket Vision is a deep learning application that identifies cricket batting shots from uploaded video clips and generates short, rule-based cricket commentary.

## Features

- Cricket video upload and prediction
- Five cricket shot categories
- Automatic rule-based commentary
- Prediction confidence and inference time
- LSTM, GRU and BiLSTM comparison
- Model evaluation dashboard

## Supported Classes

1. Cover Drive
2. Pull
3. Straight Drive
4. Sweep
5. Square Cut

## Model Architecture

The selected architecture combines:

- EfficientNetV2-S for spatial feature extraction.
- LSTM for temporal sequence learning.
- Fully connected layer for final classification.

## Dataset

The project uses a subset of the CricShot10k dataset.

| Detail | Value |
|---|---:|
| Total clips | 3,871 |
| Classes | 5 |
| Frames per video | 15 |
| Frame resolution | 224 × 224 |
| Training samples | 2,489 |
| Validation samples | 566 |
| Test samples | 816 |

Group-wise splitting by source video ID was used to reduce data leakage.

## Model Comparison

| Model | Validation Accuracy | Test Accuracy | Macro F1 |
|---|---:|---:|---:|
| EfficientNetV2-S + LSTM | 69.43% | 72.92% | 72.01% |
| EfficientNetV2-S + GRU | 72.44% | 71.57% | 70.88% |
| EfficientNetV2-S + BiLSTM | 71.02% | 71.08% | 69.84% |

LSTM was selected based on its highest test accuracy and macro F1.

## Selected Model Metrics

| Metric | Score |
|---|---:|
| Accuracy | 72.92% |
| Macro Precision | 74.20% |
| Macro Recall | 70.79% |
| Macro F1 | 72.01% |
| Weighted F1 | 73.07% |

## Technology Stack

Python, PyTorch, Torchvision, EfficientNetV2-S, LSTM, GRU, BiLSTM, OpenCV, NumPy, Pandas and Streamlit.

## Installation

Clone the repository:

    git clone YOUR_GITHUB_REPOSITORY_URL
    cd cricket_vision

Install dependencies:

    pip install -r requirements.txt

Run the application:

    streamlit run app.py

## Limitations

- Currently supports five predefined shot categories.
- Audio is not processed.
- Unknown shots are still classified into one of the five existing classes.
- Commentary is rule-based.
- Confidence scores are not necessarily calibrated probabilities.

## Future Improvements

- Object detection and batsman tracking.
- Pose estimation.
- Audio analysis.
- Unknown-shot detection.
- Improved temporal modeling.
- Richer cricket commentary.

## Academic Project

Developed as an academic project exploring computer vision, deep learning and temporal modeling for cricket video understanding.
