# Driver Drowsiness Detection System

## Project Overview
This project detects possible driver drowsiness using eye images and a trained deep learning model. It uses a webcam to detect the face and eyes and displays AWAKE or DROWSY predictions.

## Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

## Dataset
The model was trained using the MRL Eye Dataset. Dataset files are not included in this repository.

## Model Performance
- Test accuracy: 84%
- Awake recall: 77%
- Sleepy recall: 92%

Results are from the current test evaluation and may vary with model versions and conditions.

## Project Structure
- `app.py` — webcam detection application
- `src/train.py` — model training
- `src/evaluate.py` — model evaluation
- `src/preprocessing.py` — preprocessing code
- `results/` — evaluation report and graphs
- `requirements.txt` — Python dependencies

## Setup
1. Create and activate a Python virtual environment.
2. Install dependencies:
   `pip install -r requirements.txt`
3. Place the trained model at:
   `models/augmented_drowsiness_model.keras`
4. Run the application:
   `python app.py`

## Important Note
This is an academic prototype, not a certified driver-safety system. Do not rely on it to make real-world driving decisions.
