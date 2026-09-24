"""
utils/emotion_labels.py
Single source of truth for the 7 emotion classes.
"""

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

# Restrained palette colors matching design specification
EMOTION_COLORS = {
    "Angry": "#4B5563",
    "Disgust": "#4B5563",
    "Fear": "#4B5563",
    "Happy": "#1E3A8A",      # Active accent
    "Neutral": "#1E3A8A",    # Active accent
    "Sad": "#4B5563",
    "Surprise": "#1E3A8A",   # Active accent
}

NUM_CLASSES = len(EMOTIONS)
IMG_SIZE = 48
