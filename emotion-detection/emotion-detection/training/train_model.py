"""
training/train_model.py

Trains the emotion-recognition CNN on dataset/train and dataset/test,
saves the trained model to model/emotion_model.keras, and writes
accuracy/loss curves + a confusion matrix to results/.

Expected dataset layout (already created for you):

    dataset/
      train/
        Angry/      *.jpg or *.png
        Disgust/
        Fear/
        Happy/
        Neutral/
        Sad/
        Surprise/
      test/
        Angry/
        ... (same 7 folders)

A good starting dataset is FER-2013 (Kaggle) — download it and drop the
images into the matching class folders above.

Run:
    python training/train_model.py
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.emotion_labels import EMOTIONS, IMG_SIZE, NUM_CLASSES  # noqa: E402

# ----------------------------- Paths & hyperparameters -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")
MODEL_DIR = os.path.join(BASE_DIR, "model")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_PATH = os.path.join(MODEL_DIR, "emotion_model.keras")

BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 0.001

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ----------------------------- 1. Data generators -----------------------------
def build_generators():
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=EMOTIONS,
        shuffle=True,
    )
    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=EMOTIONS,
        shuffle=False,
    )
    return train_gen, test_gen


# ----------------------------- 2. Model architecture -----------------------------
def build_model():
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation="relu", padding="same"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(NUM_CLASSES, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ----------------------------- 3. Training -----------------------------
def train():
    train_gen, test_gen = build_generators()
    model = build_model()
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True),
    ]

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=test_gen,
        callbacks=callbacks,
    )

    model.save(MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")

    plot_history(history)
    evaluate(model, test_gen)


# ----------------------------- 4. Plots -----------------------------
def plot_history(history):
    # Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "accuracy.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Loss
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "loss.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved accuracy.png and loss.png to {RESULTS_DIR}")


# ----------------------------- 5. Evaluation -----------------------------
def evaluate(model, test_gen):
    test_gen.reset()
    preds = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes

    report = classification_report(y_true, y_pred, target_names=EMOTIONS)
    print("\nClassification Report:\n", report)
    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion_matrix.png to {RESULTS_DIR}")


if __name__ == "__main__":
    train()
