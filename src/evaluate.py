
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "MRL-Eye-Dataset"
MODEL_PATH = BASE_DIR / "models" / "augmented_drowsiness_model.keras"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_names=["awake", "sleepy"],
    shuffle=False,
    label_mode="binary",
)

model = tf.keras.models.load_model(MODEL_PATH)

y_true = np.concatenate([
    labels.numpy().reshape(-1)
    for _, labels in test_ds
]).astype(int)

probabilities = model.predict(test_ds, verbose=1).reshape(-1)
y_pred = (probabilities >= 0.5).astype(int)

print("\nClass mapping: 0 = awake, 1 = sleepy")
print("\nConfusion matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification report:")
print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["awake", "sleepy"],
    zero_division=0,
))
