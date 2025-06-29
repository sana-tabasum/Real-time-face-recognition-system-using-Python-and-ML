try:
    import cv2
    import torch
    import numpy as np
    import os
    import pandas as pd
    from PIL import Image
    from datetime import datetime
    from facenet_pytorch import MTCNN, InceptionResnetV1
    from sklearn.metrics.pairwise import cosine_similarity
    from skimage.feature import local_binary_pattern
except ModuleNotFoundError as e:
    print(f"[ERROR] Missing module: {e.name}. Please install it with 'pip install {e.name}'")
    raise SystemExit(1)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize models
mtcnn = MTCNN(keep_all=False, device=device)
arcface = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Transform function
def transform(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# Histogram equalization
def enhance_lighting(image):
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

# Get face embedding
def get_face_embedding(img):
    try:
        face = mtcnn(img)
        if face is not None:
            face = face.unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = arcface(face)
            return embedding.cpu().numpy()
    except Exception as e:
        print(f"[ERROR] Failed to get embedding: {e}")
    return None

# LBP-based anti-spoofing
def detect_spoof_lbp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    uniform_score = hist[0] / np.sum(hist)
    return uniform_score < 0.85

# Load known faces
def load_face_database(folder='face_database'):
    face_db = {}
    if not os.path.exists(folder):
        print(f"[ERROR] Face database folder '{folder}' does not exist.")
        return face_db

    loaded = 0
    for file in os.listdir(folder):
        if file.lower().endswith(('jpg', 'jpeg', 'png')):
            name = os.path.splitext(file)[0]
            img_path = os.path.join(folder, file)
            try:
                img = Image.open(img_path).convert('RGB')
                emb = get_face_embedding(img)
                if emb is not None:
                    face_db[name] = emb
                    loaded += 1
            except Exception as e:
                print(f"[ERROR] Failed to load {img_path}: {e}")
    print(f"[INFO] Loaded {loaded} faces from database.")
    return face_db

# Recognize face from embedding
def recognize_face(embedding, face_db, threshold=0.6):
    best_match = "Unknown"
    best_score = 0
    for name, ref_emb in face_db.items():
        score = cosine_similarity(ref_emb, embedding)[0][0]
        if score > threshold and score > best_score:
            best_match = name
            best_score = score
    return best_match, best_score

# Logging recognized names
def log_recognition(name):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([[now, name]], columns=["Timestamp", "Name"])
    df.to_csv("recognition_log.csv", mode='a', index=False, header=not os.path.exists("recognition_log.csv"))

# Main execution
face_db = load_face_database()

if not face_db:
    print("[ERROR] No faces loaded from database. Please add images to 'face_database' folder.")
    raise SystemExit(1)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot open webcam.")
    raise SystemExit(1)

print("[INFO] Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    frame = enhance_lighting(frame)
    rgb_pil = transform(frame)
    embedding = get_face_embedding(rgb_pil)
    label = "Face Not Detected"
    color = (0, 0, 255)

    if embedding is not None:
        if detect_spoof_lbp(frame):
            name, score = recognize_face(embedding, face_db)
            if name != "Unknown":
                log_recognition(name)
                label = f"{name} ({score:.2f})"
                color = (0, 255, 0)
            else:
                label = "Unknown Face"
                color = (0, 165, 255)
        else:
            label = "Spoof Detected"
            color = (0, 0, 255)

    cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
