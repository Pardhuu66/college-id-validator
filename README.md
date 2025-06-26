# 🎓 College ID Validator

An AI-powered offline system for detecting fake, altered, or non-genuine student ID cards using image classification, OCR, face detection, and template matching — deployed via FastAPI and Docker.

---

## 📌 Features

- 📄 Image classification using MobileNetV2 (ONNX)
- 🔍 OCR text extraction via EasyOCR
- 👤 Face detection using OpenCV DNN
- 📋 Template matching for reference ID layout verification
- ⚖️ Aggregates confidence scores for final decision: Genuine, Suspicious, or Fake
- 🚀 REST API powered by FastAPI
- 📦 Packaged as a Docker container for portable, environment-independent deployment

---

## 🛠️ Tech Stack

- Python 3.11
- FastAPI, Uvicorn
- MobileNetV2 (ONNX) with ONNXRuntime
- EasyOCR, OpenCV, OpenCV DNN
- Docker
- Pytest for unit testing

---

## 📂 Project Structure

```

college-id-validator/
├── Dockerfile
├── README.md
├── requirements.txt
├── config.json
├── collegeDataSet.csv
├── college_id_classifier.onnx
├── main.py
├── face_detector.py
├── image_classifier.py
├── ocr_validator.py
├── template_validator.py
├── models/
│   ├── deploy.prototxt.txt
│   └── res10_300x300_ssd_iter_140000.caffemodel
├── templates/
│   └── (reference ID templates)
├── tests/
│   └── (unit test files)

````

---

## 📸 Example API Request

**Endpoint:** `POST /validate-id`

### 📥 Request Body

```json
{
  "user_id": "test_user",
  "image_base64": "iVBORw0KGgoAAAANSUhEUgA..."
}
````

### 📤 Sample Response

```json
{
  "user_id": "test_user",
  "validation_score": 0.89,
  "label": "genuine",
  "status": "approved"
}
```

---

## 🐳 Docker Commands

### 📦 Build Docker Image

```
docker build -t college-id-validator .
```

### 🚀 Run Docker Container

```
docker run -p 8000:8000 college-id-validator
```

### 📄 Access API Documentation

[http://localhost:8000/docs](http://localhost:8000/docs)

### 📤 Save Docker Image to .tar Archive

```
docker save -o college-id-validator.tar college-id-validator
```

### 📥 Load Docker Image from .tar Archive

```
docker load -i college-id-validator.tar
```

---

## 📊 Results

| Metric                        | Score |
| :---------------------------- | :---- |
| Image Classification Accuracy | \~96% |
| OCR Text Extraction Accuracy  | \~92% |
| Face Detection Success Rate   | \~95% |

✔️ Stable, consistent performance inside a Dockerized environment.

---

## 📚 References

* Dataset inspiration from Kaggle
* Official Documentation:

  * EasyOCR
  * OpenCV
  * Docker
* YouTube tutorials for FastAPI and Docker
* ChatGPT and Grok AI for debugging and optimization
* Notion templates and technical blogs for architecture design

---

## ✨ Author

**Pardha Sai Gudivada**

* [GitHub Profile](https://github.com/Pardhuu66)

```

---

