# Real-Time Multi-User Face Recognition System with Anti-Spoofing

This project is a robust, real-time face recognition system developed using Python and advanced machine learning algorithms. It incorporates anti-spoofing mechanisms, multi-user detection, and live webcam/video feed support. The system is built for real-world deployment in surveillance, secure access control, and attendance applications.

## 📌 Overview
- **Tech Stack**: Python, OpenCV, Facenet-PyTorch, Sklearn, PIL, Pandas
- **Framework**: Flask (for optional web interface)
- **Main Features**:
  - Real-time face detection & recognition
  - Anti-spoofing detection using LBP texture analysis
  - Embedding-based identity verification
  - Logging of recognition attempts
  - Web interface for image upload and live results

## 🧠 Key Components
- **Face Detection**: MTCNN (Multi-task Cascaded CNN)
- **Embedding Extraction**: InceptionResnetV1 (VGGFace2 pretrained)
- **Anti-Spoofing**: Local Binary Pattern (LBP)
- **Recognition**: Cosine Similarity
- **Logging**: Time-stamped CSV logs using Pandas

## 🎯 Functional Highlights
- Multi-user support with dynamic database loading
- Live webcam integration (`cv2.VideoCapture`)
- Liveness detection to avoid printed/photo/video spoof attacks
- Face log file (`recognition_log.csv`) generation
- Optionally supports a web interface via Flask

## 📂 Folder Structure
real-time-face-recognition/
│
├── face_database/ # Known user images
├── recognition_log.csv # Log of recognized faces
├── app.py # Main program
├── static/ & templates/ # Web interface (Flask)
├── README.md # This file

markdown
Copy
Edit

## 🧪 Example Use Cases
- Smart surveillance systems
- Secure login portals
- Student attendance marking
- Employee recognition at entry gates

## 🔐 Anti-Spoofing Measures
- Texture-based spoof detection (LBP)
- Optional blink detection and motion cues (future enhancement)
- Secure embedding encryption and threshold filtering

## 🚀 How to Run (CLI)
```bash
pip install -r requirements.txt
python app.py
🌐 Optional: Web Interface
Upload image → detect face → verify → display name or unknown/spoof

Uses Flask and AJAX for frontend-backend integration

🔬 Algorithms Used
MTCNN (face detection)

FaceNet / VGGFace2 (embeddings)

Cosine Similarity (classification)

LBP (anti-spoofing)

Siamese Networks, ArcFace, and Triplet Loss (designed in architecture)

📝 Future Enhancements
3D face recognition using stereo depth

Eye blink/motion detection for better spoof prevention

Cloud deployment (Firebase/AWS)

Real-time alerts or notifications
press q to quit webcam.

## 📦 Dataset Download

Due to GitHub's file size limit, the dataset is not included here.

You can download the dataset from:

👉 [Download datasets.zip from Google Drive](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset?resource=download)
