import cv2
import numpy as np
import os
from PIL import Image
# --- Configuration ---
# Paths to the pre-trained face detection model
PROTOTXT_PATH = "models/deploy.prototxt.txt"
WEIGHTS_PATH = "models/res10_300x300_ssd_iter_140000.caffemodel"

# Thresholds for validation
MIN_FACE_CONFIDENCE = 0.7  # Minimum confidence for a detected face
MIN_BLUR_SCORE = 100.0     # Minimum Laplacian variance (higher is sha   rper). Tune this!
MIN_FACE_SIZE_PIXELS = 25 # Minimum width/height of the detected face in pixels. Tune this!

# --- Helper Functions ---

def load_face_detection_model(prototxt_path, weights_path):
    """Loads the pre-trained face detection model."""
    if not os.path.exists(prototxt_path) or not os.path.exists(weights_path):
        print(f"Error: Model files not found. Ensure '{prototxt_path}' and '{weights_path}' exist.")
        '''print("You can download them from:")
        print("- Prototxt: https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/deploy.prototxt.txt")
        print("- Caffemodel: https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel")'''
        return None
    net = cv2.dnn.readNetFromCaffe(prototxt_path, weights_path)
    return net

def detect_faces(image, net, min_confidence=0.5):
    """Detects faces in an image using the provided dnn network."""
    (h, w) = image.shape[:2]
    # Preprocess the image: resize to 300x300 and normalize
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    detected_faces_info = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > min_confidence:
            # Extract coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure the bounding box is within the image dimensions
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            # Calculate area for prioritizing larger faces if multiple are found
            area = (endX - startX) * (endY - startY)
            if area > 0: # Make sure it's a valid box
                 detected_faces_info.append({
                    "box": (startX, startY, endX, endY),
                    "confidence": float(confidence),
                    "area": area
                })
    return detected_faces_info

def calculate_blurriness(image_gray):
    """Calculates the Laplacian variance, a proxy for blurriness."""
    return cv2.Laplacian(image_gray, cv2.CV_64F).var()

def validate_face_on_id(image: Image.Image, face_net=None, 
                        min_confidence=MIN_FACE_CONFIDENCE,
                        min_blur_score=MIN_BLUR_SCORE, 
                        min_face_size=MIN_FACE_SIZE_PIXELS):
    """
    Validates if a face is present and meets quality criteria in the image.
    Args:
        image: PIL Image object
    Returns:
        bool: True if a valid face is detected (genuine), False otherwise (fake)
    """
    # Convert PIL image to OpenCV format (BGR)
    image_np = np.array(image.convert("RGB"))
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Load face detection model
    face_net = load_face_detection_model(PROTOTXT_PATH, WEIGHTS_PATH)
    if face_net is None:
        print("Face detection model not loaded.")
        return False
    
    # Detect faces
    detected_faces = detect_faces(image_cv, face_net, min_confidence=MIN_FACE_CONFIDENCE)
    if not detected_faces:
        return False
    
    # Select face with largest area
    best_face_info = max(detected_faces, key=lambda x: x['area'])
    (startX, startY, endX, endY) = best_face_info["box"]
    
    # Extract face ROI
    face_roi = image_cv[startY:endY, startX:endX]
    if face_roi.size == 0:
        return False
    
    # Check face size
    face_h, face_w = face_roi.shape[:2]
    if face_w < MIN_FACE_SIZE_PIXELS or face_h < MIN_FACE_SIZE_PIXELS:
        return False
    
    # Check blurriness
    gray_face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    blur_score = calculate_blurriness(gray_face_roi)
    if blur_score < MIN_BLUR_SCORE:
        return False
    
    return True

'''def validate_face_on_id(image_path, face_net, 
                        min_face_confidence_detect=MIN_FACE_CONFIDENCE,
                        min_blur_score_thresh=MIN_BLUR_SCORE,
                        min_face_size_thresh=MIN_FACE_SIZE_PIXELS):
    """
    Validates the face on an ID card image.
    Returns: (status: "genuine" or "fake", message: reason_string)
    """
    image = cv2.imread(image_path)
    if image is None:
        return "fake", f"Could not load image from path: {image_path}"

    # 1. Detect faces
    detected_faces = detect_faces(image, face_net, min_confidence=min_face_confidence_detect)

    if not detected_faces:
        return "fake", "No human face detected with sufficient confidence."

    # If multiple faces are detected, typically the ID photo is the most prominent.
    # We'll pick the one with the largest area.
    # You might need more sophisticated logic if other faces (e.g., holograms) interfere.
    best_face_info = max(detected_faces, key=lambda x: x['area'])
    
    (startX, startY, endX, endY) = best_face_info["box"]
    face_confidence = best_face_info["confidence"]
    
    # Extract the face ROI (Region of Interest)
    face_roi = image[startY:endY, startX:endX]

    if face_roi.size == 0:
         return "fake", "Detected face ROI is empty (likely bad bounding box)."

    # 2. Check face size (resolution constraint)
    face_h, face_w = face_roi.shape[:2]
    if face_w < min_face_size_thresh or face_h < min_face_size_thresh:
        return "fake", (f"Detected face is too small ({face_w}x{face_h}px). "
                        f"Minimum required: {min_face_size_thresh}x{min_face_size_thresh}px.")

    # 3. Check face clarity (blurriness constraint)
    gray_face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    blur_score = calculate_blurriness(gray_face_roi)

    if blur_score < min_blur_score_thresh:
        return "fake", (f"Detected face is too blurry (Blur score: {blur_score:.2f}). "
                        f"Minimum required: {min_blur_score_thresh:.2f}.")

    # If all checks pass
    return "genuine", (f"Face validated. Confidence: {face_confidence:.2f}, "
                       f"Size: {face_w}x{face_h}px, Blur score: {blur_score:.2f}.")


# --- Main Execution & Single Test Case ---
if __name__ == "__main__":
    # Load the face detection model ONCE
    face_detection_net = load_face_detection_model(PROTOTXT_PATH, WEIGHTS_PATH)

    if face_detection_net is None:
        print("Exiting due to model loading failure.")
    else:
        print("Face detection model loaded successfully.")
        
        # --- Test Case ---
        # Create a dummy image or use a real ID card image (with privacy in mind)
        # For this example, let's assume you have an image 'test_id_card.jpg'
        # in the same directory as your script.
        
        # You should replace 'test_id_card.jpg' with the path to your test image.
        test_image_path = "image.jpeg" # <--- REPLACE WITH YOUR IMAGE PATH

        # Create a dummy image if test_id_card.jpg doesn't exist for testing purposes
        if not os.path.exists(test_image_path):
            print(f"Test image '{test_image_path}' not found. Creating a placeholder image.")
            print("Please replace it with a real ID card image for proper testing.")
            # Create a simple blank image for testing the flow if no image is provided
            # This will likely fail validation, which is expected for a blank image.
            dummy_img = np.zeros((600, 400, 3), dtype=np.uint8)
            cv2.putText(dummy_img, "ID Card Area", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            # You might want to add a very clear, large face to this dummy image to test a "genuine" case
            # For now, it will likely be "fake" due to no face.
            cv2.imwrite(test_image_path, dummy_img)
            print(f"Created a dummy image at '{test_image_path}'. Please replace it with a real test image.")


        print(f"\n--- Validating: {test_image_path} ---")
        status, message = validate_face_on_id(test_image_path, face_detection_net)
        print(f"Face Validation Status: {status.upper()}")
        print(f"Details: {message}")

        # Example of visualizing the detected face (optional)
        if status == "genuine" or "too blurry" in message or "too small" in message : # If a face was found
            image_to_show = cv2.imread(test_image_path)
            if image_to_show is not None:
                detections_viz = detect_faces(image_to_show, face_detection_net, MIN_FACE_CONFIDENCE)
                if detections_viz:
                    best_face_viz = max(detections_viz, key=lambda x: x['area'])
                    (startX, startY, endX, endY) = best_face_viz["box"]
                    cv2.rectangle(image_to_show, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.putText(image_to_show, f"{status}: {best_face_viz['confidence']:.2f}", (startX, startY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Resize for display if it's too large
                    max_display_height = 700
                    if image_to_show.shape[0] > max_display_height:
                        scale_ratio = max_display_height / image_to_show.shape[0]
                        width = int(image_to_show.shape[1] * scale_ratio)
                        height = int(image_to_show.shape[0] * scale_ratio)
                        image_to_show_resized = cv2.resize(image_to_show, (width, height))
                    else:
                        image_to_show_resized = image_to_show

                    # cv2.imshow("Detected Face", image_to_show_resized)
                    #print("\n(Showing image with detected face. Press any key to close.)")
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()'''