FROM python:3.11-slim

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . /app

# Verify face detection model files exist
RUN [ -f /app/models/deploy.prototxt.txt ] && \
    [ -f /app/models/res10_300x300_ssd_iter_140000.caffemodel ] && \
    [ -f /app/college_id_classifier.onnx ] && echo "Face and classifier models verified" || { echo "Model files missing"; exit 1; }

# Verify EasyOCR models exist
RUN [ -f /app/ocr_models/model/craft_mlt_25k.pth ] && \
    [ -f /app/ocr_models/model/english_g2.pth ] && echo "EasyOCR models verified" || { echo "EasyOCR model files missing"; exit 1; }

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI app via Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
