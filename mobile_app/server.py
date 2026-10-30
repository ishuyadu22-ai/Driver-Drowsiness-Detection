
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from pathlib import Path
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "augmented_drowsiness_model.keras"

model = None

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)


def get_model():
    global model
    if model is None:
        model = load_model(MODEL_PATH, compile=False)
    return model


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "running",
        "model_exists": MODEL_PATH.exists()
    })


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Image file is required"}), 400

    raw = np.frombuffer(request.files["image"].read(), np.uint8)
    frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=4,
        minSize=(50, 50)
    )

    if len(faces) == 0:
        return jsonify({
            "status": "NO_FACE",
            "message": "Face not detected. Improve lighting and face the camera."
        })

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])

    face = frame[y:y+h, x:x+w]
    face_gray = gray[y:y+h, x:x+w]

    eyes = eye_cascade.detectMultiScale(
        face_gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(10, 10)
    )

    if len(eyes) == 0:
        return jsonify({
            "status": "EYES_NOT_DETECTED",
            "message": "Eyes not detected. Improve lighting and face the camera."
        })

    eyes = sorted(
        eyes,
        key=lambda box: box[2] * box[3],
        reverse=True
    )[:2]

    scores = []

    for ex, ey, ew, eh in eyes:
        eye = face[ey:ey+eh, ex:ex+ew]
        eye = cv2.resize(eye, (128, 128))
        eye = cv2.cvtColor(eye, cv2.COLOR_BGR2RGB)
        eye = eye.astype(np.float32)

        batch = np.expand_dims(eye, axis=0)

        score = float(get_model().predict(batch, verbose=0)[0][0])
        scores.append(score)

    sleepy_score = float(np.mean(scores))
    print("Individual eye scores:", scores)
    print("Average sleepy score:", sleepy_score)
    status = "DROWSY" if sleepy_score >= 0.5 else "AWAKE"

    return jsonify({
    "status": status,
    "sleepy_probability": round(sleepy_score, 4),
    "individual_eye_scores": [round(s, 4) for s in scores],
    "eyes_detected": len(eyes),
    "message": "Prototype testing only."
})


if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5050)), debug=False)