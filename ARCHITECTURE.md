# ARCHITECTURE — Emotion AI

**Status:** Draft
**Last updated:** September 2026

---

## 1. System Overview

```
                    ┌─────────────┐
                    │    USER     │
                    └──────┬──────┘
                           │ webcam permission
                           ▼
                 ┌───────────────────┐
                 │   STREAMLIT UI    │  (app.py)
                 └─────────┬─────────┘
                           │ video frames (streamlit-webrtc)
                           ▼
                 ┌───────────────────┐
                 │  FACE DETECTION   │  OpenCV Haar Cascade
                 └─────────┬─────────┘
                           │ (x, y, w, h) box
                           ▼
                 ┌───────────────────┐
                 │  PREPROCESSING    │  crop → grayscale → 48×48 → normalize
                 └─────────┬─────────┘
                           │ tensor (1, 48, 48, 1)
                           ▼
                 ┌───────────────────┐
                 │     CNN MODEL     │  TensorFlow/Keras, loaded once & cached
                 └─────────┬─────────┘
                           │ 7 class probabilities
                           ▼
                 ┌───────────────────┐
                 │   VISUALIZATION   │  bbox overlay, bar chart, history, stats
                 └───────────────────┘
```

There is no backend server, API, or database — Streamlit's own process
handles UI, orchestration, and inference together, with all state scoped
to the browser session.

---

## 2. The Stack

| Layer | Technology | Why |
|---|---|---|
| **Language** | Python 3.x | Native ecosystem for both the ML pipeline and the web UI, no context-switching between languages |
| **Deep learning framework** | TensorFlow / Keras | Industry-standard, high-level Sequential API is fast to build and easy to explain in a viva |
| **Computer vision** | OpenCV (`opencv-python-headless`) | Face detection (Haar Cascade) and all frame-level image ops (crop, resize, color conversion) |
| **Web UI framework** | Streamlit | Turns a Python script into a shareable web app with zero HTML/CSS/JS required |
| **Real-time video** | `streamlit-webrtc` | Provides continuous, low-latency browser webcam access via WebRTC — a plain `cv2.VideoCapture` loop only works locally, this works when deployed too |
| **Video frame container** | PyAV (`av`) | Required by `streamlit-webrtc` to decode/encode video frames |
| **Data handling / display** | NumPy, Pandas | Tensor math and building the tables/charts shown in the UI |
| **Training visualization** | Matplotlib, Seaborn | Accuracy/loss curves and the confusion matrix heatmap |
| **Evaluation metrics** | scikit-learn | `classification_report`, `confusion_matrix` |

**Explicitly not used (by design):** Node.js, Express, REST API layer,
MongoDB, MySQL, or any other database/backend framework — the app has no
need to persist data beyond a single browser session, so adding server
infrastructure would only add complexity without adding capability.

### Why Streamlit over a plain HTML/CSS/JS frontend
Running a TensorFlow model directly in-browser would require converting it
to TensorFlow.js and re-implementing preprocessing in JavaScript — extra
work that adds risk without adding value for an academic deliverable.
Streamlit keeps the entire stack in Python, so the same preprocessing code
used in training (`utils/preprocessing.py`) is guaranteed to behave
identically at inference time.

---

## 3. The Data Models

This project has no relational/document database — "data models" here
means the concrete data structures that flow through the system, from
training data on disk to in-memory session state in the app.

### 3.1 Training data (on disk)

```
dataset/
├── train/
│   ├── Angry/      *.jpg | *.png
│   ├── Disgust/
│   ├── Fear/
│   ├── Happy/
│   ├── Neutral/
│   ├── Sad/
│   └── Surprise/
└── test/
    └── (same 7 folders)
```
- **Unit:** a single grayscale face image, ideally already cropped to a face
- **Label:** implicit in the folder name (Keras `flow_from_directory` infers the class from the parent directory)
- **Canonical class order:** `EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]` in `utils/emotion_labels.py` — this order must stay consistent between training and inference, since the model's output index (0–6) is meaningless without it

### 3.2 Model input tensor (runtime)

| Field | Type / Shape | Notes |
|---|---|---|
| Preprocessed face | `float32`, shape `(1, 48, 48, 1)` | Batch size 1 (single live frame), 48×48 grayscale, normalized 0–1 |

### 3.3 Model output (prediction record)

| Field | Type | Example |
|---|---|---|
| `probabilities` | `float32[7]` | `[0.02, 0.01, 0.04, 0.91, 0.01, 0.01, 0.00]` |
| `label` | `str` (argmax of the above, mapped via `EMOTIONS`) | `"Happy"` |
| `confidence` | `float` (max probability) | `0.91` |

### 3.4 In-session state (Streamlit `session_state`)

```python
history: deque[(timestamp: str, label: str, confidence: float)]
# maxlen=50 — a rolling window of the most recent detections
```

Derived, not stored separately:
- **Session summary** — a `Counter` over `history` labels, computed on
  each render (not persisted; recalculated from `history`)
- **Live probability bars** — the most recent `probabilities` array,
  held in a local variable per frame, not written to `session_state`
  (it doesn't need to survive a rerun)

### 3.5 Evaluation artifacts (on disk, post-training)

```
results/
├── accuracy.png
├── loss.png
├── confusion_matrix.png
└── classification_report.txt
```
These are static files generated once by `training/train_model.py` and
simply displayed as images/text by the "Model Performance" tab — the app
never regenerates them at runtime.

### 3.6 Model artifact

```
model/
└── emotion_model.keras
```
A single serialized Keras model file, loaded once via `@st.cache_resource`
so it isn't reloaded on every Streamlit rerun (which happens on nearly
every UI interaction).

---

## 4. Data flow summary

```
Training time:
  dataset/train, dataset/test  →  train_model.py  →  model/emotion_model.keras
                                                    →  results/*.png, *.txt

Runtime (per frame):
  webcam frame  →  face box  →  tensor (1,48,48,1)  →  probabilities[7]
                                                      →  session_state.history (append)
                                                      →  UI (bbox, bars, history table, stats chart)
```

No data ever leaves the user's browser/session — there is no network call
carrying webcam frames or predictions to an external server.
