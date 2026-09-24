"""
utils/emotion_labels.py

Single source of truth for the 7 emotion classes.
Import EMOTIONS wherever you need the class order — it MUST match the
order Keras assigns during training (alphabetical, by default, when using
flow_from_directory), so keep folder names identical to these labels.
"""

# Order matters: this must match the alphabetical folder order Keras uses
# in ImageDataGenerator.flow_from_directory (dataset/train/<label>/...)
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

EMOTION_EMOJIS = {
    "Angry": "😠",
    "Disgust": "🤢",
    "Fear": "😨",
    "Happy": "😊",
    "Neutral": "😐",
    "Sad": "😢",
    "Surprise": "😲",
}

# Used for the probability bar chart / bounding box color in the UI
EMOTION_COLORS = {
    "Angry": "#e74c3c",
    "Disgust": "#8e44ad",
    "Fear": "#34495e",
    "Happy": "#f1c40f",
    "Neutral": "#95a5a6",
    "Sad": "#3498db",
    "Surprise": "#e67e22",
}

NUM_CLASSES = len(EMOTIONS)
IMG_SIZE = 48  # 48x48 grayscale, classic FER-2013 format
