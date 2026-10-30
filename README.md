# Personalized Mobile-Based Driver Drowsiness Detection System

## Project Overview

The **Personalized Mobile-Based Driver Drowsiness Detection System** is a deep learning-based application designed to detect possible driver drowsiness using eye images captured through a camera.

The system uses a trained CNN model to classify eye conditions into two categories:

- **AWAKE** – The driver appears alert.
- **DROWSY** – The system detects signs of closed or sleepy eyes.

A mobile-friendly web interface displays real-time predictions, sleepy probability, and warning alerts to help demonstrate the detection process.

## Key Features

- Real-time face and eye detection using a camera.
- Deep learning-based Awake and Drowsy classification.
- Sleepy probability display.
- Automatic drowsiness warning after 5 consecutive DROWSY predictions.
- Beep sound alert when drowsiness is detected.
- Automatic alert reset when the system detects an awake state.
- No Face Detection status.
- Camera Start and Stop functionality.
- Mobile-friendly user interface.

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- Flask
- HTML, CSS and JavaScript
- NumPy
- Matplotlib
- Scikit-learn

## Datasets

The project uses the following publicly available datasets:

1. **MRL Eye Dataset** – Used for eye image-based model training.
2. **NTHU-DDD Dataset** – Used as an additional drowsiness detection dataset.
3. **UTA-RLDD Dataset** – Used as an additional driver drowsiness dataset.

Dataset files are stored locally and are not included in this repository because of their large size.

## Model Performance

Current evaluation results:

- Test Accuracy: 84%
- Awake Recall: 77%
- Sleepy Recall: 92%

These results are based on the current model evaluation and may vary depending on the model version and testing conditions.

## Project Structure

```text
Personalized-Mobile-Based-Driver-Drowsiness-Detection-System/
│
├── data/
│
├── models/
│   └── augmented_drowsiness_model.keras
│
├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── preprocessing.py
│   └── requirements.txt
│
├── mobile_app/
│   ├── index.html
│   └── server.py
│
├── results/
│
├── app.py
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/ishuyadu22-ai/Personalized-Mobile-Based-Driver-Drowsiness-Detection-System.git
```

### 2. Open the Project Folder

```bash
cd Personalized-Mobile-Based-Driver-Drowsiness-Detection-System
```

### 3. Create a Virtual Environment

```bash
python3 -m venv .venv
```

### 4. Activate the Environment

For macOS / Linux:

```bash
source .venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Backend Server

```bash
python3 mobile_app/server.py
```

The Flask backend will run at:

```text
http://127.0.0.1:5050
```

### 7. Run the Mobile App Interface

Open another terminal:

```bash
cd mobile_app
python3 -m http.server 8000
```

Open the following address in your browser:

```text
http://localhost:8000
```

Allow camera access and click **Start Camera** to begin detection.

## Drowsiness Alert Mechanism

The system monitors consecutive drowsiness predictions.

When 5 consecutive DROWSY predictions are detected:

1. A red DROWSINESS ALERT warning appears.
2. An audio beep alert is triggered.
3. The alert resets when the system detects an awake state.

## Project Objective

The main objective is to develop a camera-based prototype that demonstrates how deep learning and computer vision can be used to identify possible driver drowsiness and generate timely visual and audio alerts.

## Important Note

This project is developed for academic and educational purposes. It is an experimental prototype and is not a certified driver-safety system. Its predictions may be inaccurate, so it must not be relied upon for real-world driving decisions.
