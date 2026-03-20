import io
import gc
import os
import re
import cv2
import base64
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import pytesseract

app = FastAPI()

# Allow CORS so the frontend can make requests to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Lazy-loaded YOLO model ----
# Model is NOT loaded at startup to save memory.
# It is loaded on the first request instead.
_model = None


def get_yolo_model():
    """Lazy-load the YOLO model on first use."""
    global _model
    if _model is None:
        from ultralytics import YOLO
        try:
            _model = YOLO('./runs/detect/train9/weights/best.pt')
            gc.collect()
            print("✅ YOLO model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
    return _model


def preprocess_plate_for_ocr(plate_crop):
    """
    Preprocess a license plate crop for better Tesseract OCR accuracy.
    Converts to grayscale, applies thresholding, and resizes.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_RGB2GRAY)

    # Resize for better OCR (scale up small plates)
    h, w = gray.shape
    if w < 200:
        scale = 200 / w
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Apply bilateral filter to reduce noise while keeping edges
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # Apply adaptive thresholding for better text extraction
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    return thresh


def extract_plate_text(plate_crop):
    """
    Extract license plate text from a cropped plate image using Tesseract OCR.
    Much lighter than EasyOCR (~20MB vs ~200MB RAM).
    """
    processed = preprocess_plate_for_ocr(plate_crop)

    # Configure Tesseract for license plate reading
    # --psm 7: Treat the image as a single text line
    # -c tessedit_char_whitelist: Only allow uppercase letters and digits
    config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

    raw_text = pytesseract.image_to_string(processed, config=config).strip()

    # Also try with --psm 8 (single word) as fallback
    if len(raw_text) < 4:
        config_alt = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        raw_text_alt = pytesseract.image_to_string(processed, config=config_alt).strip()
        if len(raw_text_alt) > len(raw_text):
            raw_text = raw_text_alt

    # Clean the text
    cleaned = re.sub(r'[^A-Za-z0-9]', '', raw_text).upper()

    if not cleaned:
        return ""

    # Try to find typical Indian plate format: 2 letters, 1-2 digits, 1-3 letters, 4 digits
    plate_match = re.search(r'([A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4})', cleaned)

    if plate_match:
        raw_plate = plate_match.group(1)
        best_text = re.sub(r'([A-Z]+)([0-9]+)', r'\1 \2 ', raw_plate)
        best_text = re.sub(r'([0-9]+)([A-Z]+)', r'\1 \2 ', best_text).strip()
        return " ".join(best_text.split())
    else:
        # Fallback: format the cleaned string
        best_text = re.sub(r'([A-Z]+)([0-9]+)', r'\1 \2 ', cleaned)
        best_text = re.sub(r'([0-9]+)([A-Z]+)', r'\1 \2 ', best_text).strip()
        best_text = " ".join(best_text.split())

        # If too long, keep the last 4 blocks (typical plate: TN 09 BY 9726)
        words = best_text.split()
        if len(words) > 4:
            best_text = " ".join(words[-4:])

        return best_text


def get_base64_image(img_arr):
    """Convert an RGB numpy array to a base64-encoded JPEG string."""
    img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buffer).decode('utf-8')


@app.get("/health")
async def health_check():
    """Lightweight health endpoint that doesn't load models."""
    return {"status": "ok"}


@app.post("/predict")
async def predict_license_plate(file: UploadFile = File(...)):
    model = get_yolo_model()

    if not model:
        return {"error": "YOLO model not loaded", "detections": []}

    try:
        # Read the image from the uploaded file
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")

        # Limit input image size to save memory
        max_dim = 1280
        if max(pil_img.size) > max_dim:
            pil_img.thumbnail((max_dim, max_dim), Image.LANCZOS)

        img_np = np.array(pil_img)

        # Run inference using YOLOv8
        results = model.predict(pil_img, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                # Crop the license plate region
                plate_crop = img_np[y1:y2, x1:x2]

                # Perform OCR using Tesseract (lightweight!)
                best_text = extract_plate_text(plate_crop)

                base64_crop = get_base64_image(plate_crop)

                detections.append({
                    "text": best_text.strip(),
                    "confidence": conf,
                    "image": base64_crop
                })

        # Sort by confidence
        detections.sort(key=lambda x: x["confidence"], reverse=True)

        # Clean up to free memory
        del img_np, pil_img, results, contents
        gc.collect()

        return {"detections": detections}
    except Exception as e:
        import traceback
        traceback.print_exc()
        gc.collect()
        return {"error": str(e), "detections": []}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
