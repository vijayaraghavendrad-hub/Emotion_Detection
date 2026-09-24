"""
app.py — Emotion AI: Real-Time Facial Emotion Recognition
Forwarder/entrypoint to ensure compatibility from emotion-detection directory.
"""
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
nested_app = os.path.join(CURRENT_DIR, "emotion-detection", "app.py")
root_app = os.path.join(CURRENT_DIR, "..", "app.py")

if os.path.exists(nested_app):
    with open(nested_app, encoding="utf-8") as f:
        code = f.read()
    exec(code)
elif os.path.exists(root_app):
    with open(root_app, encoding="utf-8") as f:
        code = f.read()
    exec(code)
