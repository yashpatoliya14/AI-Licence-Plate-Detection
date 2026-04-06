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

app = FastAPI()

# Allow CORS so the frontend can make requests to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "https://ai-licence-plate-detection-1.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Lazy-loaded globals ----
_model = None
_reader = None

# Store EasyOCR recognition weights in a persistent folder.
# This prevents re-downloading models after each restart.
EASYOCR_MODEL_DIR = os.path.join(os.path.dirname(__file__), "easyocr_models")
os.makedirs(EASYOCR_MODEL_DIR, exist_ok=True)

# If models were downloaded previously to the old (often ephemeral) directory,
# copy them into the persistent cache folder to avoid re-downloading.
_OLD_EASYOCR_MODEL_DIR = "/tmp/easyocr_models"
try:
    if os.path.isdir(_OLD_EASYOCR_MODEL_DIR) and os.path.exists(_OLD_EASYOCR_MODEL_DIR):
        # Only copy when the new cache directory is empty.
        if not any(os.scandir(EASYOCR_MODEL_DIR)):
            import shutil
            shutil.copytree(_OLD_EASYOCR_MODEL_DIR, EASYOCR_MODEL_DIR, dirs_exist_ok=True)
except Exception:
    # Non-fatal: if copy fails, EasyOCR will download as needed.
    pass


def get_yolo_model():
    """Lazy-load the YOLO model on first use."""
    global _model
    if _model is None:
        from ultralytics import YOLO
        try:
            _model = YOLO('./runs/detect/train9/weights/best.pt')
            
            gc.collect()
            print("YOLO model loaded successfully")
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
    return _model


def get_ocr_reader():
    """Lazy-load EasyOCR reader on first use."""
    global _reader
    if _reader is None:
        import easyocr
        try:
            _reader = easyocr.Reader(
                ['en'],
                gpu=False,
                model_storage_directory=EASYOCR_MODEL_DIR,
                download_enabled=True,
            )
            gc.collect()
            print("EasyOCR reader initialized successfully")
        except Exception as e:
            print(f"Failed to initialize EasyOCR: {e}")
    return _reader


def get_base64_image(img_arr):
    """Convert an RGB numpy array to a base64-encoded JPEG string."""
    img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buffer).decode('utf-8')


def extract_plate_text(plate_crop, reader):
    """Extract and format license plate text from a cropped plate image."""
    ocr_result = reader.readtext(plate_crop)

    if not ocr_result:
        return ""

    raw_text = " ".join([text for (bbox, text, prob) in ocr_result])
    cleaned = re.sub(r'[^A-Za-z0-9]', '', raw_text).upper()

    parts = raw_text.split()
    valid_parts = []
    for p in parts:
        p_clean = re.sub(r'[^A-Za-z0-9]', '', p).upper()
        if len(p_clean) >= 2 or p_clean.isdigit():
            valid_parts.append(p_clean)

    cleaned_str = "".join(valid_parts)

    # Try to match Indian plate format: 2 letters, 1-2 digits, 1-3 letters, 4 digits
    plate_match = re.search(r'([A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4})', cleaned_str)

    if plate_match:
        raw_plate = plate_match.group(1)
        best_text = re.sub(r'([A-Z]+)([0-9]+)', r'\1 \2 ', raw_plate)
        best_text = re.sub(r'([0-9]+)([A-Z]+)', r'\1 \2 ', best_text).strip()
        return " ".join(best_text.split())
    else:
        best_text = re.sub(r'([A-Z]+)([0-9]+)', r'\1 \2 ', cleaned_str)
        best_text = re.sub(r'([0-9]+)([A-Z]+)', r'\1 \2 ', best_text).strip()
        best_text = " ".join(best_text.split())

        words = best_text.split()
        if len(words) > 4:
            best_text = " ".join(words[-4:])

        return best_text


@app.get("/health")
async def health_check():
    """Lightweight health endpoint that doesn't load models."""
    return {"status": "ok"}


@app.post("/predict")
async def predict_license_plate(file: UploadFile = File(...)):
    model = get_yolo_model()
    reader = get_ocr_reader()

    if not model or not reader:
        return {"error": "Model or OCR not correctly loaded", "detections": []}

    try:
        # Read the image
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")

        # Cap image size to save memory and avoid OOM (502 Bad Gateway)
        max_dim = 640
        if max(pil_img.size) > max_dim:
            pil_img.thumbnail((max_dim, max_dim), Image.LANCZOS)

        img_np = np.array(pil_img)

        # Run YOLO inference
        results = model.predict(pil_img, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                plate_crop = img_np[y1:y2, x1:x2]
                best_text = extract_plate_text(plate_crop, reader)
                base64_crop = get_base64_image(plate_crop)

                detections.append({
                    "text": best_text.strip(),
                    "confidence": conf,
                    "image": base64_crop
                })

        detections.sort(key=lambda x: x["confidence"], reverse=True)

        # Free memory
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
