# AI Licence Plate Number Detection (ALPR)

A full‑stack **Automatic Licence Plate Recognition (ALPR)** project:

- **Detection**: YOLO (Ultralytics) detects licence plate bounding boxes.
- **Recognition (OCR)**: EasyOCR extracts plate text from each detected crop.
- **Backend**: FastAPI (`DL/app.py`) exposes a simple REST API.
- **Frontend**: React + Vite (`frontend/`) provides a modern upload-and-scan UI.

> **Note:** This project is for learning/demo purposes. Accuracy depends on training data, camera angle, lighting, motion blur, and plate format.

## Project Structure

```
AI Licence Plate Number Detection/
├── DL/                       # FastAPI backend + ML assets
│   ├── app.py                # API + YOLO + EasyOCR pipeline
│   ├── requirements.txt      # Backend deps
│   ├── runs/.../best.pt      # YOLO weights (your trained model)
│   ├── easyocr_models/       # EasyOCR cached weights (created at runtime)
│   ├── Dockerfile            # Container build for deployment
│   └── Project1.ipynb        # Training/experiments notebook
├── frontend/                 # React (Vite) frontend
│   ├── src/App.jsx           # UI + calls /predict
│   └── .env                  # VITE_BACKEND_URL
├── run_backend.bat           # Windows helper to start backend
└── run_frontend.bat          # Windows helper to start frontend
```

## Requirements

- **Python** 3.10+ (3.11 recommended for deployment)
- **Node.js** 18+ and npm

## Run Locally (Windows)

### Backend (FastAPI)

From the project root:

```bash
cd DL
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Or simply double‑click:
- `run_backend.bat`

Health check:
- `GET http://localhost:8000/health` → `{"status":"ok"}`

### Frontend (React + Vite)

```bash
cd frontend
npm install
```

Create/update `frontend/.env`:

```env
VITE_BACKEND_URL=http://localhost:8000
```

Start dev server:

```bash
npm run dev
```

Or double‑click:
- `run_frontend.bat`

## API

### `GET /health`

Returns a lightweight status without loading models.

### `POST /predict`

Accepts `multipart/form-data` with a single field:
- **file**: image file (`.jpg`, `.png`, etc.)

Example using curl:

```bash
curl -X POST "http://localhost:8000/predict" ^
  -H "accept: application/json" ^
  -F "file=@test.jpg"
```

Response shape:

```json
{
  "detections": [
    {
      "text": "MH 12 AB 1234",
      "confidence": 0.87,
      "bbox": [x1, y1, x2, y2]
    }
  ]
}
```

## Configuration (Memory/Speed Tuning)

`DL/app.py` supports these environment variables:

- **MAX_IMG_DIM** (default `512`): max image side length before inference
- **YOLO_IMGSZ** (default `416`): YOLO inference image size
- **YOLO_CONF** (default `0.35`): confidence threshold
- **YOLO_MAX_DET** (default `3`): maximum detections per image

These defaults are chosen to reduce RAM usage on small instances.

## Why EasyOCR “downloads a recognition model”

EasyOCR uses pretrained OCR weights. If they are not already present, it downloads them once and stores them in:
- `DL/easyocr_models/` (persistent cache folder)

## Deployment Notes (Render / low-RAM instances)

This stack (**PyTorch + YOLO + EasyOCR**) is memory-heavy. On **512MB** instances, you may hit out-of-memory errors.

Best options:
- Use a higher-RAM plan, or
- Deploy frontend only and run backend locally, or
- Replace OCR/detector with lighter models.

The included `DL/Dockerfile` installs **CPU-only** PyTorch to keep the image smaller than CUDA builds.

