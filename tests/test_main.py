import sys
import os
# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "P:\Turtil-project\college-id-validator")))

import pytest
from main import app
from fastapi.testclient import TestClient
import base64
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, filename="tests/test_results.log", filemode="w",
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

client = TestClient(app)

# Helper function to encode image to base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Test cases with manual image path entry
def test_clear_college_id():
    # Manually enter the path to a clear college ID image
    img = "tests/original_1.png"  # Update this path
    if not os.path.exists(img):
        pytest.skip(f"Clear ID image not found at {img}")
    image_base64 = image_to_base64(img)
    response = client.post("/validate-id", json={"user_id": "test_user", "image_base64": image_base64})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "genuine"
    assert data["status"] == "approved"
    assert data["validation_score"] >= 0.85
    logger.info(f"Clear ID Test - Score: {data['validation_score']}, Label: {data['label']}, Status: {data['status']}")

def test_fake_template():
    # Manually enter the path to a fake template image
    img = "tests/fake1.jpg"  # Update this path to your fake image
    if not os.path.exists(img):
        pytest.skip(f"Fake template image not found at {img}")
    image_base64 = image_to_base64(img)
    response = client.post("/validate-id", json={"user_id": "test_user_fake", "image_base64": image_base64})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "fake"
    assert data["status"] == "rejected"
    assert data["validation_score"] < 0.6
    logger.info(f"Fake Template Test - Score: {data['validation_score']}, Label: {data['label']}, Status: {data['status']}")

def test_cropped_image():
    # Manually enter the path to a cropped/screenshot image
    img = "tests/suspicious.jpg"  # Update this path to your cropped image (corrected typo)
    if not os.path.exists(img):
        pytest.skip(f"Cropped image not found at {img}")
    image_base64 = image_to_base64(img)
    response = client.post("/validate-id", json={"user_id": "test_user_cropped", "image_base64": image_base64})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "suspicious"
    assert data["status"] == "manual_review"
    assert 0.6 <= data["validation_score"] < 0.85
    logger.info(f"Cropped Image Test - Score: {data['validation_score']}, Label: {data['label']}, Status: {data['status']}")

def test_poor_ocr():
    # Manually enter the path to a poor OCR image
    img = "tests/suspicious.jpg"  # Update this path to your blurred image
    if not os.path.exists(img):
        pytest.skip(f"Poor OCR image not found at {img}")
    image_base64 = image_to_base64(img)
    response = client.post("/validate-id", json={"user_id": "test_user_blur", "image_base64": image_base64})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] in ["suspicious", "fake"]

    assert data["status"] == "manual_review"
    assert 0.6 <= data["validation_score"] < 0.85
    logger.info(f"Poor OCR Test - Score: {data['validation_score']}, Label: {data['label']}, Status: {data['status']}")

def test_non_id_image():
    # Manually enter the path to a non-ID image
    img = "tests/fake2.jpg"  # Update this path to your non-ID image
    if not os.path.exists(img):
        pytest.skip(f"Non-ID image not found at {img}")
    image_base64 = image_to_base64(img)
    response = client.post("/validate-id", json={"user_id": "test_user_nonid", "image_base64": image_base64})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "fake"
    assert data["status"] == "rejected"
    assert data["validation_score"] < 0.6
    logger.info(f"Non-ID Image Test - Score: {data['validation_score']}, Label: {data['label']}, Status: {data['status']}")

if __name__ == "__main__":
    pytest.main(["-v", __file__])