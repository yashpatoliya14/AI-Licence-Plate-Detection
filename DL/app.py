import io
import cv2
import base64
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import easyocr
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

# Load YOLO model
# Assuming the file 'yolo26n.pt' is your trained YOLO model for license plates
try:
    model = YOLO('./runs/detect/train9/weights/best.pt')
except Exception as e:
    print(f"Failed to load YOLO model: {e}")
    model = None

# Initialize EasyOCR reader
try:
    reader = easyocr.Reader(['en'], gpu=False)
except Exception as e:
    print(f"Failed to initialize EasyOCR: {e}")
    reader = None

def get_base64_image(img_arr):
    # Convert RGB array to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', img_bgr)
    return base64.b64encode(buffer).decode('utf-8')

@app.post("/predict")
async def predict_license_plate(file: UploadFile = File(...)):
    if not model or not reader:
        return {"error": "Model or OCR not correctly loaded", "detections": []}

    try:
        # Read the image from the uploaded file
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(pil_img)
        
        # Run inference using YOLOv8
        results = model.predict(pil_img)
        
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
                    import re
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
                        if len(words) > 4: # Typically plates are 4 blocks: TN 09 BY 9726
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
        
        return {"detections": detections}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e), "detections": []}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
