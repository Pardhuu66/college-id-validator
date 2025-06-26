import easyocr

# Download models (this will download to ~/.EasyOCR by default)
reader = easyocr.Reader(['en'], download_enabled=True)

print("EasyOCR models downloaded successfully.")
