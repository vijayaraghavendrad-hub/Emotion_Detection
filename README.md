# 🧠 Emotion AI — Real-Time Facial Emotion Recognition

A CNN-based, backend-free web app that detects a user's facial emotion in
real time from their webcam, built with Python + TensorFlow/Keras + OpenCV + Streamlit.

---

## 1. Project structure

```
emotion-detection/
│
├── app.py                     # Streamlit web app (live detection UI)
│
├── model/
│   └── emotion_model.keras    # produced by training/train_model.py
│
├── dataset/
│   ├── train/<7 emotion folders>/   # put training images here
│   └── test/<7 emotion folders>/    # put test/validation images here
│
├── training/
│   └── train_model.py         # builds, trains, evaluates the CNN
│
├── utils/
│   ├── emotion_labels.py      # class names, emojis, colors, IMG_SIZE
│   └── preprocessing.py       # face detection + frame → tensor pipeline
│
├── results/                   # accuracy.png, loss.png, confusion_matrix.png
├── assets/                    # logo/branding images for the app (optional)
├── requirements.txt
└── README.md
```

All seven class folders (`Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise`)
already exist under `dataset/train/` and `dataset/test/` — just drop images in.

---

## 2. Step-by-step build order

### Step 1 — Environment
```bash
cd emotion-detection
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2 — Get a dataset
Use **FER-2013** (Kaggle, ~35k labeled 48×48 grayscale faces across the same
7 classes) or your own webcam-captured images. Sort images into
`dataset/train/<Emotion>/` and `dataset/test/<Emotion>/` to match the folder
names exactly (case-sensitive).

Recommended split if you're building the dataset yourself: 80% train / 10%
validation / 10% test. In this project, "validation" is the `test/` folder
passed to Keras during training (`validation_data=test_gen`); if you want a
true held-out test set as well, keep a third folder aside and only evaluate
on it after training is finished.

### Step 3 — Train the model
```bash
python training/train_model.py
```
This will:
- Load images via `ImageDataGenerator` (with augmentation: rotation, shift, zoom, flip)
- Build the CNN (architecture below)
- Train for up to 40 epochs with early stopping on validation loss
- Save the best model to `model/emotion_model.keras`
- Save `accuracy.png`, `loss.png`, `confusion_matrix.png`, and
  `classification_report.txt` to `results/`

### Step 4 — Run the app
```bash
streamlit run app.py
```
Open the local URL Streamlit prints. Grant webcam permission. The **Live
Detection** tab starts streaming predictions; **Model Performance** shows
your training graphs; **About** explains the project.

---

## 3. CNN architecture

```
Input (48×48×1)
   ↓
Conv2D(32, 3×3, relu) → MaxPooling(2×2)
   ↓
Conv2D(64, 3×3, relu) → MaxPooling(2×2)
   ↓
Conv2D(128, 3×3, relu) → MaxPooling(2×2)
   ↓
Flatten
   ↓
Dense(128, relu)
   ↓
Dropout(0.5)
   ↓
Dense(7, softmax)
```

## 4. Key parameters (viva-ready reference)

| Parameter | Value | Why |
|---|---|---|
| Input size | 48×48×1 (grayscale) | Standard FER-2013 format, small enough for fast CPU training |
| Classes | 7 | Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise |
| Filters | 32 → 64 → 128 | Deeper layers learn increasingly complex facial features |
| Kernel size | 3×3 | Standard for capturing local features without excessive params |
| Pool size | 2×2 | Halves spatial dimensions, reduces computation |
| Hidden activation | ReLU | Avoids vanishing gradients, fast to compute |
| Output activation | Softmax | Converts logits to a probability distribution across 7 classes |
| Dropout | 0.5 | Regularization — reduces overfitting before the final layer |
| Optimizer | Adam | Adaptive learning rate, converges reliably for image CNNs |
| Learning rate | 0.001 | Adam's typical default starting point |
| Loss | Categorical cross-entropy | Standard for multi-class, one-hot labeled classification |
| Batch size | 32 | Good balance of gradient stability and memory use |
| Epochs | up to 40 (early stopping) | Stops automatically once validation loss stops improving |

## 5. Real-time inference pipeline

```
Webcam frame (BGR)
   ↓
Haar Cascade face detection (OpenCV)
   ↓
Crop face → grayscale → resize to 48×48 → normalize (0–255 → 0–1)
   ↓
Reshape to (1, 48, 48, 1)
   ↓
model.predict() → 7 probabilities
   ↓
argmax → label + confidence → drawn on frame + shown in UI
```

This exact pipeline lives in `utils/preprocessing.py` and is shared by
`app.py`, so training-time and inference-time preprocessing never drift
apart.

## 6. What to include in your report / demo

- Training vs validation **accuracy curve** (`results/accuracy.png`)
- Training vs validation **loss curve** (`results/loss.png`)
- **Confusion matrix** (`results/confusion_matrix.png`) — discuss which
  emotions get confused (commonly Sad ↔ Neutral, Fear ↔ Surprise)
- **Classification report** (precision/recall/F1 per class)
- Live demo screenshot/recording of the Streamlit app detecting your own face

## 7. Extending it further (optional, no backend needed)

- Multi-face detection: loop over all boxes from `detect_faces()`, not just `faces[0]`
- Export `history` to CSV via a Streamlit download button
- Swap Haar Cascade for a more accurate detector (e.g., MediaPipe Face Detection)
- Quantize the model (`TFLite`) for faster inference on low-power devices
- Deploy free on **Streamlit Community Cloud** by pushing this repo to GitHub

## 8. Requirements

See `requirements.txt`. No MongoDB, MySQL, Node.js, Express, or REST API
needed for this version — Streamlit handles the UI, and everything else
runs in-process.
