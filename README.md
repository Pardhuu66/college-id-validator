# 🎓 College ID Validator

An AI-powered offline system for detecting fake, altered, or non-genuine student ID cards using image classification, OCR, face detection, and template matching — deployed via FastAPI and Docker.

---

## 📌 Features

- 📄 Image classification using MobileNetV2 (ONNX)
- 🔍 OCR text extraction using EasyOCR
- 👤 Face detection with OpenCV DNN
- 📋 Template matching for layout verification
- ⚖️ Score aggregation for final decision: Genuine, Suspicious, or Fake
- 🚀 FastAPI-based REST API for easy integration
- 📦 Dockerized for portable, environment-independent deployment

---

## 🛠️ Tech Stack

- **Python 3.11**
- **FastAPI**, **Uvicorn**
- **MobileNetV2 (ONNX)** with **ONNXRuntime**
- **EasyOCR**, **OpenCV**, **OpenCV DNN**
- **Docker**
- **Pytest** for testing

---

## 📂 Project Structure

college-id-validator/
├── main.py
├── ocr_validator.py
├── image_classifier.py
├── templates/
│ └── (reference ID templates)
├── models/
│ └── college_id_classifier.onnx
├── Dockerfile
├── requirements.txt
├── test_main.py
└── README.md

yaml
Copy
Edit

---

## 📸 Example API Request

**Endpoint:** `POST /validate-id`

**Request Body:**

json
{
  "user_id": "test_user",
  "image_base64": "iVBORw0KGgoAAAANSUhEUgA..."
}
Sample Response:

{
  "user_id": "test_user",
  "validation_score": 0.89,
  "label": "genuine",
  "status": "approved"
}
🐳 Docker Commands
Build Image:

bash
Copy
Edit
docker build -t college-id-validator .
Run Container:

bash
Copy
Edit
docker run -p 8000:8000 college-id-validator
Access API Docs:
http://localhost:8000/docs

Save Docker Image to .tar:

bash
Copy
Edit
docker save -o college-id-validator.tar college-id-validator
Load Docker Image from .tar:

bash
Copy
Edit
docker load -i college-id-validator.tar
📊 Results
Image Classification Accuracy: ~90%

OCR Text Extraction Accuracy: ~92%

Face Detection Success Rate: ~95%

Stable performance within Docker containerized environment.

📚 References
Dataset Inspiration from Kaggle

Official Documentation for:

EasyOCR

OpenCV

Docker

Video Tutorials for Docker and FastAPI on YouTube

ChatGPT and Grok AI for code debugging and optimization suggestions.

Notion and technical blogs for selecting model architectures and design patterns.

✨ Author
Pardhu Gudivada
