"""
utils/preprocessing.py

Everything needed to turn a raw webcam frame into a CNN-ready tensor:
    frame -> detect face -> crop -> resize(48x48) -> grayscale -> normalize

Compatible with OpenCV 4.x and 5.x.
"""

import cv2
import numpy as np

from utils.emotion_labels import IMG_SIZE

# OpenCV 5.0 removed CascadeClassifier from the top-level cv2 module.
# We guard against this so the app loads cleanly on either version.
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
    """
    Detect faces in a BGR frame.

    Returns a list of (x, y, w, h) bounding boxes sorted largest-first.
    Returns an empty list if CascadeClassifier is unavailable (OpenCV 5.x).
    """
    if _FACE_CASCADE is None:
        return []
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )
    # detectMultiScale returns a tuple when nothing is found on some versions
    if not hasattr(faces, '__len__') or len(faces) == 0:
        return []
    return sorted(faces, key=lambda box: box[2] * box[3], reverse=True)


def preprocess_face(frame_bgr, box):
    """
    Crop a face out of the frame using (x, y, w, h) and turn it into the
    exact tensor shape the CNN expects: (1, 48, 48, 1), float32, 0-1 range.
    """
    x, y, w, h = box
    face = frame_bgr[y : y + h, x : x + w]

    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    normalized = resized.astype("float32") / 255.0
    tensor = np.expand_dims(normalized, axis=-1)  # (48, 48, 1)
    tensor = np.expand_dims(tensor, axis=0)  # (1, 48, 48, 1)
    return tensor


def draw_result(frame_bgr, box, label, confidence, color=(0, 200, 0)):
    """Draw the bounding box + label on the frame for display (in-place)."""
    x, y, w, h = box
    cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)
    text = f"{label} {confidence * 100:.1f}%"
    cv2.putText(
        frame_bgr,
        text,
        (x, max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )
    return frame_bgr
