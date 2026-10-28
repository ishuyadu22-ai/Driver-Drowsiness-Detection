
from pathlib import Path
import cv2
import tensorflow as tf

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "augmented_drowsiness_model.keras"

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Face and eye detectors
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open camera.")
    raise SystemExit

print("Camera started. Press q to quit.")

while True:
    success, frame = camera.read()
    if not success:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        face_gray = gray[y:y+h, x:x+w]
        eyes = eye_detector.detectMultiScale(
            face_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
        )

        for (ex, ey, ew, eh) in eyes:
            # Extract eye image from the original color frame
            eye_img = frame[y+ey:y+ey+eh, x+ex:x+ex+ew]

            if eye_img.size == 0:
                continue

            # Prepare image for the trained model
            eye_img = cv2.resize(eye_img, (128, 128))
            eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB)
            eye_img = tf.convert_to_tensor(eye_img, dtype=tf.float32)
            eye_img = tf.expand_dims(eye_img, axis=0)

            # Model predicts 0 = awake, 1 = sleepy
            probability = float(model.predict(eye_img, verbose=0)[0][0])
            status = "DROWSY" if probability >= 0.5 else "AWAKE"
            color = (0, 0, 255) if status == "DROWSY" else (0, 255, 0)

            cv2.rectangle(
                frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), color, 2
            )
            cv2.putText(
                frame, f"{status} ({probability:.2f})",
                (x+ex, max(20, y+ey-10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
            )

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

