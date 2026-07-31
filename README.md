# GoDezk Desk Management & Occupancy System

YOLO-based person detection & desk occupancy service — high-performance microservice API, zero-lag multi-camera client, and evaluation tools.

---

## 🚀 Quick Start

### 1. Installation

```bash
# Activate virtual environment
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start the FastAPI Detection Microservice API

```bash
python src/main.py
```
*API runs on `http://localhost:8000`*

### 3. Launch Zero-Lag Live Camera Stream Client

```bash
# Webcam
python live_cam.py --url 0

# IP Camera / RTSP Stream
python live_cam.py --url "rtsp://admin:Admin%40123@106.51.106.43:40002/video/live?channel=1&subtype=0"
```

---

## 🧪 Testing Utilities

### Image Occupancy Detection
```bash
# Single image test
python test_image.py --image path/to/image.jpg

# Directory batch test
python test_image.py --dir path/to/images/

# Test using running API service
python test_image.py --image path/to/image.jpg --server http://localhost:8000/detect
```

### Video File Processing
```bash
# Process video file and generate annotated output MP4
python test_video.py --video path/to/video.mp4
```

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health status, model load status, timestamp |
| `/alive` | GET | Liveness probe (200 OK or 503 Service Unavailable) |
| `/ready` | GET | Readiness probe (200 OK or 503 Service Unavailable) |
| `/detect` | POST | Accepts binary JPEG or JSON base64 frame; returns person count & bounding boxes |
| `/metrics` | GET | Prometheus performance & alert metrics |

### Request Example (`POST /detect`)
**Header:** `Content-Type: image/jpeg`  
**Body:** Raw JPEG bytes

OR JSON payload:
```json
{
  "image_b64": "<base64_encoded_jpeg>",
  "camera_id": "desk_cam_01"
}
```

---

## ⚙️ Environment Variables (`.env`)

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `models/yolo11s.pt` | Path to YOLO model weight file |
| `CONFIDENCE_THRESHOLD` | `0.50` | Detection confidence threshold |
| `MAX_PEOPLE` | `5` | Occupancy alert limit threshold |
| `IMG_SIZE` | `640` | Model inference image dimension |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `ALERT_URL` | `""` | Optional webhook URL for occupancy breach alerts |
| `TEMPORAL_WINDOW_SIZE` | `5` | Rolling frame buffer size for edge smoothing |
| `TEMPORAL_MIN_HITS` | `2` | Minimum positive frames required for alert |
| `CAMERA_SOURCES` | `0` | Default camera RTSP/webcam URL(s) |

---

## 🐳 Docker Deployment

```bash
docker build -t desk-management-system .
docker run -p 8000:8000 desk-management-system
```
