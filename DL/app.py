import io
import gc
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Lazy-loaded globals ----
# Models are NOT loaded at startup to save memory.
# They are loaded on the first request instead.
_model = None
_reader = None


def get_yolo_model():
    """Lazy-load the YOLO model on first use."""
    global _model
    if _model is None:
        from ultralytics import YOLO
        try:
            _model = YOLO('./runs/detect/train9/weights/best.pt')
            gc.collect()  # free any transient allocations from loading
            print("✅ YOLO model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
    return _model


def get_ocr_reader():
    """Lazy-load the EasyOCR reader on first use."""
    global _reader
    if _reader is None:
        import easyocr
        try:
            _reader = easyocr.Reader(
                ['en'],
                gpu=False,
                model_storage_directory='/tmp/easyocr_models',
                download_enabled=True,
            )
            gc.collect()  # free any transient allocations from loading
            print("✅ EasyOCR reader initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize EasyOCR: {e}")
    return _reader


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
    reader = get_ocr_reader()

    if not model or not reader:
        return {"error": "Model or OCR not correctly loaded", "detections": []}

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
        # If there are detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                # Crop the license plate region
                plate_crop = img_np[y1:y2, x1:x2]

                # Perform OCR on the cropped plate
                ocr_result = reader.readtext(plate_crop)

                best_text = ""
                # Get the highest confidence text or concatenate if needed
                if ocr_result:
                    # Often EasyOCR returns multiple pieces of text; we can take the highest confidence
                    # or concatenate them. Assuming license plate is a single prominent text segment.
                    raw_text = " ".join([text for (bbox, text, prob) in ocr_result])
                    # Clean the text to keep only uppercase letters and numbers
                    cleaned = re.sub(r'[^A-Za-z0-9]', '', raw_text).upper()

                    # Fix anomalies common in OCR like O -> 0, etc for typical Indian plate format
                    # If it starts with state code (2 letters) then digits, try extracting just the plate portion
                    # In this case just look for the longest sequence of valid plate-like characters

                    # Instead of forcing T0ra, we just extract chunks that look like they belong to a plate
                    # Drop obvious noise strings that are less than 2 chars before filtering
                    parts = raw_text.split()
                    valid_parts = []
                    for p in parts:
                        p_clean = re.sub(r'[^A-Za-z0-9]', '', p).upper()
                        # If a part is purely tiny noise (like 't0ra') which usually gets squashed, let's keep it but handle formatting
                        # 't0ra' gets cleaned to 'T0RA'. We can regex filter to find typical plate formats if needed.
                        if len(p_clean) >= 2 or p_clean.isdigit():
                            valid_parts.append(p_clean)

                    cleaned_str = "".join(valid_parts)

                    # Use regex to find the most probable license plate structure within the string:
                    # Look for 2 letters, 1-2 digits, 1-3 letters, 4 digits pattern anywhere
                    plate_match = re.search(r'([A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4})', cleaned_str)

                    if plate_match:
                        # Found an exact plate match, format it nicely
                        raw_plate = plate_match.group(1)
                        best_text = re.sub(r'([A-Z]+)([0-9]+)', r'\1 \2 ', raw_plate)
                        best_text = re.sub(r'([0-9]+)([A-Z]+)', r'\1 \2 ', best_text).strip()
                        best_text = " ".join(best_text.split())
                    else:
                        # Fallback heuristic: Try to clean up small noise prefixes like T0RA
                        # Drop leading characters until we see a typical state code (2 letters followed by 1-2 digits)
                        # or just apply the formatting to the entire cleaned string
                        best_text = re.sub(r'([A-Z]+)([0-9]+)', r'\1 \2 ', cleaned_str)
                        best_text = re.sub(r'([0-9]+)([A-Z]+)', r'\1 \2 ', best_text).strip()
                        best_text = " ".join(best_text.split())

                        # Extra cleanup: If it's too long, it probably has junk at the start
                        # Example: T0RATNO9BY9726 -> T0RA TN O9 BY 9726 -> we want to drop T0RA
                        # We can look for actual state code at index > 0.
                        words = best_text.split()
                        if len(words) > 4:  # Typically plates are 4 blocks: TN 09 BY 9726
                            # Keep the last 4 blocks as the most likely actual plate
                            best_text = " ".join(words[-4:])

                base64_crop = get_base64_image(plate_crop)

                detections.append({
                    "text": best_text.strip(),
                    "confidence": conf,
                    "image": base64_crop
                })

        # Sort by confidence so the highest confidence detection is first
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
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
