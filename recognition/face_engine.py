import cv2
import numpy as np
import pickle
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier

BASE = Path(__file__).resolve().parent.parent
DATASET = BASE / "dataset"
MODEL_FILE = BASE / "models" / "knn_model.pkl"
LABELS_FILE = BASE / "models" / "labels.pkl"
CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(CASCADE)

# CLAHE for better contrast in low light
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

FACE_SIZE = 100  # face image size for training/recognition


def preprocess_face(gray_face):
    """Apply CLAHE + Gaussian blur + normalize for better recognition."""
    face = cv2.resize(gray_face, (FACE_SIZE, FACE_SIZE))
    face = clahe.apply(face)
    face = cv2.GaussianBlur(face, (3, 3), 0)
    face = face.astype(np.float32) / 255.0
    return face


def detect_faces(frame):
    """Detect faces with optimized parameters."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=6,
        minSize=(80, 80),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    return faces, gray


def train_model():
    """Train KNN with preprocessed face data and distance-weighted voting."""
    faces, labels = [], []
    label_map = {}
    DATASET.mkdir(parents=True, exist_ok=True)

    idx = 0
    for person_dir in sorted(DATASET.iterdir()):
        if not person_dir.is_dir():
            continue
        imgs = list(person_dir.glob("*.jpg"))
        if len(imgs) == 0:
            continue  # skip students with no face data
        label_map[idx] = person_dir.name
        for img_file in imgs:
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            processed = preprocess_face(img)
            faces.append(processed.flatten())
            labels.append(idx)
        idx += 1

    if len(faces) == 0:
        return False, "No face data found in dataset."

    X = np.array(faces)
    y = np.array(labels)

    # Use distance-weighted KNN for much better accuracy
    k = min(7, len(X))
    model = KNeighborsClassifier(
        n_neighbors=k,
        weights='distance',     # closer neighbors have more influence
        metric='euclidean',
    )
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
    """Recognize with preprocessing and stricter confidence check."""
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
    processed = preprocess_face(gray).flatten().reshape(1, -1)

    # Get distances to nearest neighbors for confidence check
    distances, indices = model.kneighbors(processed)
    avg_dist = distances[0].mean()

    pred = model.predict(processed)[0]
    proba = model.predict_proba(processed).max()
    name = label_map.get(pred, "Unknown")

    # If average distance is too high, face is not a good match
    # This prevents recognizing strangers as known students
    if avg_dist > 22.0:
        return "Unknown", round(proba * 100, 1)

    return name, round(proba * 100, 1)
