
#  Real Time Indian Sign Language Recognition Based on Neural Network Architecture

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.9-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red?logo=opencv)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 About The Project

A **Real-Time Indian Sign Language (ISL) Recognition System** that detects hand gestures from a live webcam feed and converts them into **text and speech** using **MediaPipe**, **Neural Network**, and **pyttsx3**.

The system recognizes **35 gestures** — **26 alphabets (A–Z)** and **9 numbers (1–9)** — from the Indian Sign Language and speaks out the detected sign in real time.

---
##  Features

- ✅ **Real-time** hand gesture detection using webcam
- ✅ Recognizes **35 ISL gestures** (A–Z + 1–9)
- ✅ **Text display** on screen with detected sign
- ✅ **Speech output** using pyttsx3 (text-to-speech)
- ✅ **21 hand landmarks** extracted using MediaPipe
- ✅ **Special phrase mapping** for numbers (1, 2, 3)
- ✅ **Normalized keypoint** based classification
- ✅ Works in **real-time** with live webcam feed

---

## How It Works

```
Webcam captures live video frame
            ↓
MediaPipe detects hand in frame
(Hand Detection Neural Network)
            ↓
21 hand landmarks extracted
(x, y coordinates per finger joint)
            ↓
Landmarks converted to relative
& normalized coordinates (42 values)
            ↓
Feed-Forward Neural Network classifies
gesture into one of 35 ISL signs
            ↓
Predicted sign displayed on screen
+ spoken aloud using pyttsx3
```

---

## System Architecture

| Stage | Technology | What it Does |
|-------|-----------|--------------|
| **Hand Detection** | MediaPipe CNN | Detects hand in webcam frame |
| **Landmark Extraction** | MediaPipe | Extracts 21 hand keypoints |
| **Preprocessing** | NumPy | Normalizes coordinates |
| **Classification** | Neural Network (Keras) | Predicts sign from landmarks |
| **Text Display** | OpenCV | Shows predicted sign on screen |
| **Speech Output** | pyttsx3 | Speaks the detected sign |

---

## Model Details

| Property | Details |
|----------|---------|
| **Model Type** | Feed-Forward Neural Network |
| **Input** | 42 normalized keypoint values (21 landmarks × 2) |
| **Output** | 35 classes (A–Z + 1–9) |
| **Framework** | TensorFlow / Keras |
| **Saved As** | `model.h5` |
| **Dataset** | Custom ISL image dataset (1199 images/class) |

---

## Gestures Recognized

### Alphabets (A–Z)
```
A  B  C  D  E  F  G  H  I  J
K  L  M  N  O  P  Q  R  S  T
U  V  W  X  Y  Z
```

### Numbers (1–9)
```
1  2  3  4  5  6  7  8  9
```

### Special Phrase Mapping
| Gesture | Phrase Spoken |
|---------|--------------|
| **1** | "Where is the Bus Stand" |
| **2** | "Where is the Canteen" |
| **3** | "Where is the Address" |

---

## 🗂️ Project Structure

```
sign-language-recognition/
│
├── maincode.py                    # Main real-time detection script
├── dataset_keypoint_generation.py # Extract keypoints from images
├── ISL_classifier.ipynb           # Model training notebook
├── model.h5                       # Trained Neural Network model
├── keypoint.csv                   # Extracted keypoints dataset
│
├── allGestures.png                # All 35 ISL gestures reference
├── example1.png                   # Live detection example — Sign B
├── example2.png                   # Live detection example — Sign C
├── process.png                    # System architecture diagram
├── signstotrain.png               # Signs used for training
│
└── images/
    └── data/
        └── A/ B/ C/ ... Z/ 1/ ... 9/   # Training images (1199 per class)
```

---

## Tech Stack

| Category | Technology |
|----------|------------|
| **Language** | Python 3.10 |
| **Hand Detection** | MediaPipe |
| **Computer Vision** | OpenCV |
| **Deep Learning** | TensorFlow / Keras |
| **Data Processing** | NumPy, Pandas |
| **Text to Speech** | pyttsx3 |
| **Model Format** | HDF5 (.h5) |
| **IDE** | Jupyter Notebook / VS Code |

---

## Installation & Setup

### Step 1 — Clone the Repository
```bash
git clone https://github.com/yourusername/sign-language-recognition.git
cd sign-language-recognition
```

### Step 2 — Install Dependencies
```bash
pip install opencv-python mediapipe tensorflow numpy pandas pyttsx3
```

### Step 3 — Run Real-Time Detection
```bash
python maincode.py
```
> 📷 Make sure your **webcam is connected**
> Press **ESC** to exit

---

## Requirements

```
opencv-python
mediapipe
tensorflow
keras
numpy
pandas
pyttsx3
```

Install all:
```bash
pip install opencv-python mediapipe tensorflow numpy pandas pyttsx3
```

---

## How Dataset Was Generated

1. **Collected** ISL hand gesture images (1199 images per class)
2. **Processed** each image using MediaPipe to extract 21 hand landmarks
3. **Converted** landmarks to **relative coordinates** (subtract base point)
4. **Normalized** all values by dividing by max absolute value
5. **Saved** as `keypoint.csv` with label + 42 keypoint values

```python
# Each row in keypoint.csv:
# [label, x0, y0, x1, y1, ..., x20, y20]  → 42 values
```

---

## Model Training

The model was trained in `ISL_classifier.ipynb`:

- **Input shape:** 42 (21 landmarks × x,y coordinates)
- **Output:** 35 classes (A–Z + 1–9)
- **Architecture:** Feed-Forward Neural Network
- **Training data:** keypoint.csv (35 × 1199 = 41,965 samples)

---

## Dataset Details

| Property | Details |
|----------|---------|
| **Type** | Custom ISL Image Dataset |
| **Classes** | 35 (26 alphabets + 9 numbers) |
| **Images per class** | 1,199 |
| **Total images** | ~41,965 |
| **Keypoints per image** | 21 landmarks (42 values) |

---

## Real-Time Detection Flow

```python
# 1. Capture webcam frame
cap = cv2.VideoCapture(0)

# 2. Detect hand landmarks using MediaPipe
results = hands.process(image)

# 3. Extract & normalize 21 keypoints → 42 values
pre_processed = pre_process_landmark(landmark_list)

# 4. Predict using Neural Network
predictions = model.predict(df)
label = alphabet[np.argmax(predictions)]

# 5. Display on screen + speak
cv2.putText(image, label, (50,50), ...)
engine.say(label)
```

---

## Code Files Explained

| File | Purpose |
|------|---------|
| `maincode.py` | Main script — webcam → detect → predict → speak |
| `dataset_keypoint_generation.py` | Extract keypoints from training images |
| `ISL_classifier.ipynb` | Train Neural Network on keypoint.csv |
| `model.h5` | Saved trained model |
| `keypoint.csv` | Generated keypoint dataset |

---

## ⚠️ Notes

- Press **ESC** key to exit the application
- Ensure good **lighting** for better hand detection
- Keep hand clearly visible in **webcam frame**
- Works best with **plain background**


