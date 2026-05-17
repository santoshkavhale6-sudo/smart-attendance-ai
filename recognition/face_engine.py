import cv2
import numpy as np
import os, pickle
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier

BASE = Path(__file__).resolve().parent.parent
DATASET = BASE / "dataset"
MODEL_FILE = BASE / "models" / "knn_model.pkl"
LABELS_FILE = BASE / "models" / "labels.pkl"
CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(CASCADE)

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))
    return faces, gray

def train_model():
    faces, labels = [], []
    label_map = {}
    DATASET.mkdir(parents=True, exist_ok=True)
    for idx, person_dir in enumerate(sorted(DATASET.iterdir())):
        if not person_dir.is_dir():
            continue
        label_map[idx] = person_dir.name
        for img_file in person_dir.glob("*.jpg"):
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (100, 100))
            faces.append(img.flatten())
            labels.append(idx)
    if len(faces) == 0:
        return False, "No face data found in dataset."
    X = np.array(faces)
    y = np.array(labels)
    k = min(5, len(X))
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X, y)
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    with open(LABELS_FILE, "wb") as f:
        pickle.dump(label_map, f)
    return True, f"Trained on {len(X)} images, {len(label_map)} students."

def load_model():
    if not MODEL_FILE.exists() or not LABELS_FILE.exists():
        return None, None
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    with open(LABELS_FILE, "rb") as f:
        label_map = pickle.load(f)
    return model, label_map

def recognize_face(face_roi, model, label_map):
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
    gray = cv2.resize(gray, (100, 100)).flatten().reshape(1, -1)
    pred = model.predict(gray)[0]
    proba = model.predict_proba(gray).max()
    name = label_map.get(pred, "Unknown")
    return name, round(proba * 100, 1)
