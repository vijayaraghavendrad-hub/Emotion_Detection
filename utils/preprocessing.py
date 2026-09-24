"""
utils/preprocessing.py
Preprocesses webcam frames into tensors for emotion classification.
Compatible with OpenCV 4.x and 5.x.
"""

import cv2
import numpy as np

try:
    from utils.emotion_labels import IMG_SIZE
except ImportError:
    IMG_SIZE = 48

# OpenCV 5.0 removed CascadeClassifier from the top-level module.
# Attempt to load it; if unavailable fall back to a stub so the rest
# of the app still loads without crashing.
_FACE_CASCADE = None
try:
    if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data"):
        import os as _os
        _cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if _os.path.exists(_cascade_path):
            _FACE_CASCADE = cv2.CascadeClassifier(_cascade_path)
except Exception:
    _FACE_CASCADE = None

def detect_faces(frame_bgr):
    """Detect faces using Haar Cascade (4.x) or return empty list (5.x fallback)."""
    if _FACE_CASCADE is None:
        return []
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )
    if not isinstance(faces, (list, tuple)) and len(faces) == 0:
        return []
    return sorted(faces, key=lambda box: box[2] * box[3], reverse=True)


def preprocess_face(frame_bgr, box):
    """Crop, grayscale, resize to 48x48, normalize 0-1, return (1, 48, 48, 1) tensor."""
    x, y, w, h = box
    face = frame_bgr[y : y + h, x : x + w]
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    normalized = resized.astype("float32") / 255.0
    tensor = np.expand_dims(normalized, axis=-1)
    return np.expand_dims(tensor, axis=0)

def draw_result(frame_bgr, box, label, confidence, color=(30, 58, 138)):
    """Draw clean bounding box and label in understated accent color."""
    x, y, w, h = box
    cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)
    text = f"{label} {confidence * 100:.1f}%"
    cv2.putText(
        frame_bgr,
        text,
        (x, max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )
    return frame_bgr
