from PIL import Image, ImageFilter
import easyocr

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Load image
img = Image.open("tests/blurred_test11.jpg").convert("RGB")

# Detect text regions
results = reader.readtext("tests/blurred_test11.jpg")

# Blur text regions
for (bbox, text, prob) in results:
    (top_left, top_right, bottom_right, bottom_left) = bbox
    x = int(top_left[0])
    y = int(top_left[1])
    w = int(top_right[0] - top_left[0])
    h = int(bottom_left[1] - top_left[1])

    region = img.crop((x, y, x + w, y + h))
    blurred_region = region.filter(ImageFilter.GaussianBlur(radius=12))
    img.paste(blurred_region, (x, y))

# Save blurred image
img.save("tests/blurred_test_easyocr.jpg")
print("Blurred all detected text regions using EasyOCR.")
