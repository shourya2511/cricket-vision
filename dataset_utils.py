
# Paste all your dataset_utils code below this line

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# ==========================================
# 1. CONFIGURATION
# ==========================================

NUM_FRAMES = 15
IMAGE_SIZE = 224

LABEL_MAP = {
    "Cover Drive": 0,
    "Pull": 1,
    "Straight Drive": 2,
    "Sweep": 3,
    "Square Cut": 4
}

NUM_CLASSES = len(LABEL_MAP)


# ==========================================
# 2. IMAGE TRANSFORM
# ==========================================

image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# 3. FRAME EXTRACTION
# ==========================================

def extract_frames(video_path, num_frames=NUM_FRAMES):

    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        raise ValueError(f"Cannot read video: {video_path}")

    frame_indices = np.linspace(
        0, total_frames - 1, num_frames, dtype=int
    )

    frames = []

    for idx in frame_indices:

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))

        ret, frame = cap.read()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    cap.release()

    if len(frames) != num_frames:
        raise ValueError(
            f"Expected {num_frames} frames, got {len(frames)}: {video_path}"
        )

    return frames


# ==========================================
# 4. VIDEO PREPROCESSING
# ==========================================

def preprocess_video(video_path):

    frames = extract_frames(video_path)

    processed_frames = []

    for frame in frames:

        image = Image.fromarray(frame)

        image = image_transform(image)

        processed_frames.append(image)

    video_tensor = torch.stack(processed_frames)

    return video_tensor


# ==========================================
# 5. CUSTOM DATASET CLASS
# ==========================================

class CricketDataset(Dataset):

    def __init__(self, dataframe):

        self.dataframe = dataframe.reset_index(drop=True)

    def __len__(self):

        return len(self.dataframe)

    def __getitem__(self, idx):

      row = self.dataframe.iloc[idx]

      video_path = row["video_path"]

      label = LABEL_MAP[row["label"]]

      video = preprocess_video(video_path)

      label = torch.tensor(label, dtype=torch.long)

      return video, label

