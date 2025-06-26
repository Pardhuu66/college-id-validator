import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os
import json

# Directory for template images
TEMPLATE_DIR = "templates"

# Load approved colleges dynamically from config.json
def load_approved_colleges(config_path="config.json"):
    try: 
        with open(config_path, "r") as f:
            config = json.load(f)
        return config.get("approved_colleges", [])
    except Exception as e:
        print(f"Failed to load approved colleges: {e}")
        return []

APPROVED_COLLEGES = load_approved_colleges()

# Template matching function
def run_template_check(image: np.ndarray) -> dict:
    try:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray_image = cv2.equalizeHist(gray_image)  # Normalize lighting
        best_score = 0.0
        best_template = None

        for college in APPROVED_COLLEGES:
            template_filename = f"{college.replace(' ', '_')}_screenshot.jpg"
            template_path = os.path.join(TEMPLATE_DIR, template_filename)

            if not os.path.exists(template_path):
                print(f"Template not found: {template_path}")
                continue

            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                print(f"Failed to load template: {template_path}")
                continue

            # Resize to match input image size
            if template.shape != gray_image.shape:
                template = cv2.resize(template, (gray_image.shape[1], gray_image.shape[0]))
            
            template = cv2.equalizeHist(template)  # Normalize lighting

            # Compute SSIM score
            score, _ = ssim(gray_image, template, full=True)
            print(f"Checking template: {template_filename}, SSIM score: {score:.4f}")

            if score > best_score:
                best_score = score
                best_template = college

        if best_score == 0.0 and best_template is None:
            return {
                "score": 0.5,
                "details": "No templates available for comparison"
            }

        # Scale SSIM to (0–1)
        best_score = min(best_score * 2, 1.0)  # Normalize SSIM to be more forgiving
        details = (
            f"Best match: {best_template if best_template else 'None'}, "
            f"Score: {best_score:.4f}"
        )

        return {
            "score": best_score,
            "details": details
        }

    except Exception as e:
        print(f"Template check failed: {str(e)}")
        return {
            "score": 0.5,
            "details": f"Template check failed: {str(e)}"
        }
    

'''
import base64
import io
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import onnxruntime as ort
import tensorflow as tf
from PIL import Image
import logging
import json
from typing import Literal
from ocr_validator import validate_id_card_fields
from face_detector import validate_face_on_id, load_face_detection_model
from template_validator import run_template_check  # Added import

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="College ID Card Validator",
    description="Validates college ID cards using classifier, OCR, face detection, and template matching.",
)

# CORS middleware
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8080",
    "http://localhost:5173",
    "http://127.0.0.1:5500",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load config
try:
    with open("config.json", "r") as f:
        config = json.load(f)
except Exception as e:
    logger.error(f"Error loading config: {e}")
    raise

# Load class indices
try:
    with open("class_indices.json", "r") as f:
        class_indices = json.load(f)
    logger.info(f"Class Indices: {class_indices}")
except FileNotFoundError:
    class_indices = {'genuine': 1, 'fake': 0}  # Default assumption
    logger.warning("class_indices.json not found. Assuming genuine=1, fake=0")

# Constants
IMG_HEIGHT, IMG_WIDTH = 224, 224
MODEL_PATH = "college_id_classifier.onnx"

# Load ONNX model
try:
    classifier_session = ort.InferenceSession(MODEL_PATH)
    classifier_input_name = classifier_session.get_inputs()[0].name
    logger.info("✅ ONNX Classifier model loaded")
except Exception as e:
    logger.error(f"Error loading ONNX model: {e}")
    raise

# Load face detection model
face_net = load_face_detection_model("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

# Pydantic models
class IDCardRequest(BaseModel):
    user_id: str
    image_base64: str = Field(..., description="Base64 encoded ID card image.")

class ValidationResponse(BaseModel):
    user_id: str
    validation_score: float = Field(..., ge=0, le=1)
    label: Literal["genuine", "suspicious", "fake"]
    status: Literal["approved", "manual_review", "rejected"]
    reason: str
    threshold: float

# Helper function
def base64_to_pil_image(base64_string: str) -> Image.Image:
    """Decodes a base64 string to a PIL image."""
    try:
        if "," in base64_string:
            base64_string = base64_string.split(',')[1]
        img_bytes = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return image
    except Exception as e:
        logger.error(f"Error decoding base64 string: {e}")
        raise HTTPException(status_code=400, detail="Invalid base64 string or image format.")

# Validation modules
def run_classifier_check(image: Image.Image) -> dict:
    """Runs ONNX classifier and returns score and details."""
    try:
        image = image.resize((IMG_WIDTH, IMG_HEIGHT))
        image_np = np.array(image).astype(np.float32)
        image_np = np.expand_dims(image_np, axis=0)
        image_np = tf.keras.applications.efficientnet.preprocess_input(image_np)
        outputs = classifier_session.run(None, {classifier_input_name: image_np})
        raw_score = outputs[0][0][0]
        # Adjust score based on class indices
        score = raw_score if class_indices.get('genuine') == 1 else 1.0 - raw_score
        details = f"Classifier score: {score:.4f}"
        logger.debug(f"Classifier raw score: {raw_score:.4f}, Adjusted score: {score:.4f}")
        return {"score": score, "details": details}
    except Exception as e:
        logger.error(f"Classifier error: {e}")
        return {"score": 0.0, "details": f"Classifier failed: {str(e)}"}

def run_ocr_check(image_data: bytes, approved_colleges: list) -> dict:
    """Runs OCR validation and returns score and details."""
    try:
        ocr_results = validate_id_card_fields(image_data, approved_colleges)
        ocr_status = ocr_results["overall_status"]
        ocr_reasons = ocr_results["reasons"]
        score = 1.0 if ocr_status == "ACCEPTED" else 0.0
        details = f"OCR status: {ocr_status}, Reasons: {', '.join(ocr_reasons) if ocr_reasons else 'None'}"
        logger.debug(details)
        return {"score": score, "details": details}
    except Exception as e:
        logger.error(f"OCR error: {e}")
        return {"score": 0.0, "details": f"OCR failed: {str(e)}"}

def run_face_check(image: Image.Image) -> dict:
    """Runs face detection and returns score and details."""
    try:
        photo_present = validate_face_on_id(image, face_net=face_net)
        score = 1.0 if photo_present else 0.0
        details = "Face detected" if photo_present else "No face detected"
        logger.debug(details)
        return {"score": score, "details": details}
    except Exception as e:
        logger.error(f"Face detection error: {e}")
        return {"score": 0.0, "details": f"Face detection failed: {str(e)}"}

# API endpoint
@app.post("/validate-id", response_model=ValidationResponse)
async def validate_id_card(request: IDCardRequest):
    """Validates an ID card using classifier, OCR, face detection, and template matching."""
    # Thresholds and weights
    APPROVE_THRESHOLD = 0.85
    REJECT_THRESHOLD = 0.6
    WEIGHTS = {
        "classifier": 0.4,  # Most important
        "ocr": 0.3,        # Second
        "face": 0.1,       # Third
        "template": 0.2    # Added for template matching
    }

    # Decode image
    image = base64_to_pil_image(request.image_base64)
    
    # Convert image to bytes for OCR
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    image_data = img_byte_arr.getvalue()

    # Run validation modules
    classifier_result = run_classifier_check(image)
    ocr_result = run_ocr_check(image_data, config["approved_colleges"])
    face_result = run_face_check(image)
    template_result = run_template_check(np.array(image))  # Added template check

    # Calculate weighted score
    final_score = (
        classifier_result["score"] * WEIGHTS["classifier"] +
        ocr_result["score"] * WEIGHTS["ocr"] +
        face_result["score"] * WEIGHTS["face"] +
        template_result["score"] * WEIGHTS["template"]  # Include template score
    )

    # Decision logic
    reasons = []
    if classifier_result["score"] < 0.7:
        reasons.append(f"Low classifier confidence ({classifier_result['details']})")
    if ocr_result["score"] < 0.6:
        reasons.append(f"OCR validation failed ({ocr_result['details']})")
    if face_result["score"] < 0.6:
        reasons.append(f"Face check failed ({face_result['details']})")
    if template_result["score"] < 0.6:  # Added template reason
        reasons.append(f"Template mismatch ({template_result['details']})")

    if final_score >= APPROVE_THRESHOLD:
        status = "approved"
        label = "genuine"
        final_reason = "All checks passed with high confidence."
    elif final_score < REJECT_THRESHOLD:
        status = "rejected"
        label = "fake"
        final_reason = " and ".join(reasons) if reasons else "Low overall confidence."
    else:
        status = "manual_review"
        label = "suspicious"
        final_reason = " and ".join(reasons) if reasons else "Moderate confidence requires review."

    # Log results
    logger.info(
        f"User: {request.user_id}, Score: {final_score:.4f}, Classifier: {classifier_result['score']:.4f}, "
        f"OCR: {ocr_result['score']:.4f}, Face: {face_result['score']:.4f}, Template: {template_result['score']:.4f}, "
        f"Label: {label}, Status: {status}"
    )

    # Return response
    return ValidationResponse(
        user_id=request.user_id,
        validation_score=round(final_score, 2),
        label=label,
        status=status,
        reason=final_reason,
        threshold=config["validation_threshold"]
    )

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/version")
async def version_info():
    return {"model_version": "1.0.0"}'''