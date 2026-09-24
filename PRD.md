# PRD — Emotion AI: Real-Time Facial Emotion Recognition

**Status:** Draft
**Owner:** [Your Name]
**Last updated:** September 2026

---

## 1. Summary

Emotion AI is a web application that uses a webcam feed and a Convolutional
Neural Network (CNN) to classify a user's facial expression into one of
seven emotions in real time, displaying the result with a confidence
score directly in the browser.

---

## 2. What You're Building

A **real-time facial emotion recognition web app** with three core
capabilities:

1. **Live capture & inference** — the app accesses the user's webcam,
   detects a face in each frame, and classifies it into one of 7 emotions
   (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise) using a custom-trained
   CNN, updating continuously as the video streams.
2. **Visual feedback** — the detected emotion is shown with an emoji, a
   confidence percentage, a bounding box drawn over the face, and a
   probability bar chart across all 7 classes.
3. **Session insight** — a rolling detection history and a session-level
   emotion breakdown (e.g. "62% Happy, 18% Neutral"), computed and stored
   entirely client-side/in-session — no database or backend required.

**Out of scope for v1:**
- Multi-user accounts, login, or persistent storage across sessions
- Multi-face tracking (v1 detects and classifies the largest/primary face only)
- Mobile native app (web-only, webcam-based)
- Emotion-based recommendations or downstream actions (e.g. music suggestions) — flagged as future scope, not built now

---

## 3. Who It's For

| Audience | Why they'd use it |
|---|---|
| **Primary: Evaluator/faculty reviewing the project** | Needs a working, demo-able system that clearly shows a trained deep learning model in action, along with the metrics (accuracy/loss curves, confusion matrix) that prove it was actually trained, not hardcoded. |
| **Secondary: The student presenting it (you)** | Needs a project that is buildable solo without a backend/database team, explainable in a viva (clear architecture, standard parameters), and impressive to demo live. |
| **Tertiary: Casual end user / classmate trying it** | Opens the page, allows webcam access, and immediately sees their own expression classified — no setup, no account needed. |

This is explicitly a **student mini-project / academic deliverable**, not a
production consumer product — the scope, stack, and success criteria below
are chosen accordingly.

---

## 4. Problem / Motivation

Facial emotion recognition is a well-established computer vision task that
demonstrates core deep learning concepts (CNNs, image preprocessing,
classification, model evaluation) in a way that's visual, interactive, and
easy for a non-technical evaluator to understand at a glance — making it a
strong choice for a project that needs to be both technically substantive
and demo-friendly.

---

## 5. Goals & Success Metrics

| Goal | Metric |
|---|---|
| Model correctly classifies common emotions | ≥ 65–70% test accuracy across 7 classes (FER-2013 benchmarks typically land 60–70% for a from-scratch CNN) |
| Real-time feel | Inference latency low enough that predictions visibly update as the user changes expression (no more than ~1s lag) |
| Demoable without setup friction | App runs with a single `streamlit run app.py` command, webcam prompt is the only user action needed |
| Evaluation is transparent | Accuracy curve, loss curve, and confusion matrix are generated automatically and viewable in-app |

---

## 6. Core Features (v1)

1. Webcam capture with Start/Stop control
2. Face detection per frame (Haar Cascade)
3. CNN inference → 7-class probability distribution
4. Bounding box + label overlay on video
5. Live probability bar chart
6. Session-scoped detection history (timestamped log)
7. Session-scoped emotion summary chart
8. In-app "Model Performance" tab showing training graphs + confusion matrix
9. "About" tab documenting dataset, architecture, and tech stack

---

## 7. Technical Approach

- **Model:** Custom CNN (3× Conv2D/MaxPooling blocks → Dense → Dropout → Softmax), trained on 48×48 grayscale FER-2013-style data
- **Stack:** Python, TensorFlow/Keras, OpenCV, Streamlit, streamlit-webrtc
- **No backend/database** — all state (history, stats) lives in the Streamlit session
- Full architecture, parameters, and folder structure are documented in `README.md`

---

## 8. Risks & Open Questions

- **Dataset quality/size** directly caps accuracy — FER-2013 is a known-imperfect dataset (mislabeled images, low resolution); accuracy ceiling should be set expectations accordingly.
- **Lighting/angle sensitivity** of Haar Cascade face detection may cause missed detections in a live demo — worth testing the demo environment beforehand.
- **Class imbalance** (e.g. "Disgust" is underrepresented in FER-2013) may hurt per-class recall — visible in the confusion matrix, worth calling out in the report rather than hiding.
- Open question: is a real held-out test set needed beyond train/validation for the academic deliverable, or is train/validation sufficient given time constraints?

---

## 9. Timeline (suggested)

| Phase | Task |
|---|---|
| Week 1 | Dataset collection/download, environment setup |
| Week 2 | CNN build + initial training runs, tune hyperparameters |
| Week 3 | Finalize model, generate evaluation artifacts (curves, confusion matrix) |
| Week 4 | Build Streamlit app, integrate live inference, polish UI, prepare report/demo |
