"""
GoDezk Desk Management & Occupancy System
==========================================
YOLO-based person detector & desk occupancy monitor — standalone HTTP API microservice.
Inference engine with multi-camera temporal buffer smoothing & Prometheus metrics.

Two occupancy modes:
  - Manual : MAX_PEOPLE > 0  — alert when detected persons exceed fixed limit
  - Auto   : MAX_PEOPLE = 0  — AI estimates desk zone capacity from chairs/tables in frame
"""
import os
import sys
import time
import base64
import logging
import asyncio
import concurrent.futures
from collections import deque, defaultdict
from pathlib import Path
from typing import Optional, List, Dict
from contextlib import asynccontextmanager

import numpy as np
import cv2
import requests
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# ── Config ────────────────────────────────────────────────────────────
MODEL_PATH           = os.environ.get("MODEL_PATH",           "models/yolo11s.pt")
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.50"))
MAX_PEOPLE           = int(os.environ.get("MAX_PEOPLE",        "5"))
IMG_SIZE             = int(os.environ.get("IMG_SIZE",          "640"))
LOG_LEVEL            = os.environ.get("LOG_LEVEL",            "INFO")
ALERT_URL            = os.environ.get("ALERT_URL",            "").strip()
WINDOW_SIZE          = int(os.environ.get("TEMPORAL_WINDOW_SIZE", "5"))
MIN_HITS             = int(os.environ.get("TEMPORAL_MIN_HITS",    "2"))


# MAX_PEOPLE == 0 enables AI auto-zone estimation mode
AUTO_ZONE_MODE = (MAX_PEOPLE == 0)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("desk_management")

if AUTO_ZONE_MODE:
    logger.info(
        "AUTO-ZONE MODE ENABLED — AI will estimate desk capacity per camera using chair/table detection"
    )
else:
    logger.info("MANUAL MODE — Fixed occupancy limit: %d people", MAX_PEOPLE)

# ── Metrics ───────────────────────────────────────────────────────────
detection_requests = Counter("desk_detection_requests_total", "Total detection requests")
occupancy_alerts   = Counter("desk_occupancy_alerts_total",   "Occupancy limit exceeded alerts")
detection_latency  = Histogram("desk_detection_latency_seconds", "Detection latency")

# ── App Lifespan ──────────────────────────────────────────────────────
_model       = None
_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)

@asynccontextmanager
async def lifespan(application):
    global _model
    logger.info("Loading YOLO person detection model...")
    loop   = asyncio.get_event_loop()
    _model = await loop.run_in_executor(_thread_pool, _load_model)
    logger.info("Desk Management Service ready")
    yield

app = FastAPI(title="GoDezk Desk Management System", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Temporal Buffer ────────────────────────────────────────────────────
_camera_buffers = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))

def _update_camera_buffer(camera_id: Optional[str], is_over_limit: bool) -> bool:
    if WINDOW_SIZE <= 0:
        return is_over_limit
    cid = camera_id or "default"
    buf = _camera_buffers[cid]
    buf.append(is_over_limit)
    return sum(buf) >= MIN_HITS

# ── Auto-Zone Estimator (lazy import when needed) ─────────────────────
_zone_estimator = None

def _get_zone_estimator():
    global _zone_estimator
    if _zone_estimator is None:
        from zone import ZoneCapacityEstimator
        _zone_estimator = ZoneCapacityEstimator()
    return _zone_estimator

# ── Pydantic Models ────────────────────────────────────────────────────
class BoundingBox(BaseModel):
    x1:         float
    y1:         float
    x2:         float
    y2:         float
    confidence: float
    label:      str

class DetectRequest(BaseModel):
    image_b64: str
    camera_id: Optional[str] = None
    frame_id:  Optional[str] = None

class DetectResponse(BaseModel):
    success:             bool
    persons_detected:    int
    limit_exceeded:      bool
    confirmed_crowded:   bool             # temporally smoothed alert
    status:              str              # 'normal' or 'crowded'
    max_confidence:      float
    detections:          List[BoundingBox]
    camera_id:           Optional[str] = None
    frame_id:            Optional[str] = None
    processing_time_ms:  float
    # Auto-zone fields (None when MAX_PEOPLE > 0 / manual mode)
    auto_zone_mode:      bool
    zone_capacity:       Optional[int]   = None   # estimated or configured limit
    auto_zone_capacity:  Optional[int]   = None   # alias expected by API clients
    zone_method:         Optional[str]   = None   # "manual", "chair_count", "table_area", etc.

class HealthResponse(BaseModel):
    model_config    = {"protected_namespaces": ()}
    status:          str
    model_loaded:    bool
    timestamp:       float
    auto_zone_mode:  bool
    manual_limit:    Optional[int] = None
    estimated_capacity: Optional[int] = None
    auto_zone_capacities: Dict[str, int] = {}

# ── Helpers ───────────────────────────────────────────────────────────
def _decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    return img

def _load_model():
    """Load YOLO model for CPU inference."""
    from ultralytics import YOLO

    path_to_load = MODEL_PATH
    if not os.path.exists(MODEL_PATH):
        fallback = "yolov8n.pt"
        logger.warning(
            f"Custom model '{MODEL_PATH}' not found. Falling back to '{fallback}'."
        )
        path_to_load = fallback

    model = YOLO(path_to_load)
    try:
        model.fuse()
    except Exception:
        pass

    dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    model.predict(dummy, imgsz=IMG_SIZE, conf=CONFIDENCE_THRESHOLD, classes=[0], verbose=False)

    logger.info(f"Model loaded successfully from {path_to_load}")
    return model

def _run_inference(img: np.ndarray, camera_id: Optional[str] = None):
    """
    Run YOLO inference for persons.
    In auto-zone mode, also runs zone capacity estimation.
    Returns (detections, zone_capacity, zone_method).
    """
    results = _model.predict(
        img,
        imgsz=IMG_SIZE,
        conf=CONFIDENCE_THRESHOLD,
        classes=[0],    # person only
        iou=0.45,
        max_det=20,
        verbose=False,
    )

    detections = []
    if results and len(results) > 0:
        result = results[0]
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                raw_label = str(result.names.get(cls_id, "person")).lower()
                if raw_label == "person" or cls_id == 0:
                    detections.append({
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                        "confidence": round(conf, 4),
                        "label": "person",
                    })

    # Auto-zone capacity estimation
    zone_capacity = None
    zone_method   = None

    if AUTO_ZONE_MODE:
        estimator = _get_zone_estimator()
        cid = camera_id or "default"
        zone_capacity, zone_method = estimator.get_capacity(
            cid, img, _model, detections
        )
    else:
        zone_capacity = MAX_PEOPLE
        zone_method   = "manual"

    return detections, zone_capacity, zone_method


def _send_alert(message: str):
    if not ALERT_URL:
        return
    try:
        logger.info(f"Sending alert: {message}")
        resp = requests.post(ALERT_URL, json={"message": message}, timeout=5)
        resp.raise_for_status()
        logger.info(f"Webhook alert delivered (HTTP {resp.status_code})")
    except Exception as exc:
        logger.error(f"Failed to send webhook alert: {exc}")

# ── Endpoints ─────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    capacities: Dict[str, int] = {}
    if AUTO_ZONE_MODE and _zone_estimator is not None:
        capacities = _zone_estimator.cached_capacities()

    return HealthResponse(
        status="healthy" if _model is not None else "starting",
        model_loaded=_model is not None,
        timestamp=time.time(),
        auto_zone_mode=AUTO_ZONE_MODE,
        manual_limit=MAX_PEOPLE if not AUTO_ZONE_MODE else None,
        estimated_capacity=max(capacities.values()) if capacities else None,
        auto_zone_capacities=capacities,
    )

@app.get("/alive")
async def alive():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "alive"}

@app.get("/ready")
async def ready():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}

@app.post("/detect", response_model=DetectResponse)
async def detect(request: Request):
    """Detect persons and check occupancy.
    Accepts binary JPEG (Content-Type: image/*) or JSON {image_b64, camera_id?, frame_id?}.
    When MAX_PEOPLE=0: zone capacity is auto-estimated from chairs/tables in the scene."""
    detection_requests.inc()
    start = time.time()

    content_type = request.headers.get("content-type", "application/json")
    if content_type.startswith("image/"):
        raw       = await request.body()
        img       = _decode_image(raw)
        camera_id = request.headers.get("x-camera-id")
        frame_id  = request.headers.get("x-frame-id", str(time.time()))
    else:
        body      = await request.json()
        req       = DetectRequest(**body)
        img       = _decode_image(base64.b64decode(req.image_b64))
        camera_id = req.camera_id
        frame_id  = req.frame_id or str(time.time())

    try:
        loop = asyncio.get_event_loop()
        detections, zone_capacity, zone_method = await loop.run_in_executor(
            _thread_pool, _run_inference, img, camera_id
        )

        persons_detected  = len(detections)
        effective_limit   = zone_capacity or MAX_PEOPLE
        limit_exceeded    = persons_detected > effective_limit
        buf_confirmed     = _update_camera_buffer(camera_id, limit_exceeded)
        confirmed_crowded = limit_exceeded and buf_confirmed
        max_confidence    = max((d["confidence"] for d in detections), default=0.0)
        status_str        = "crowded" if limit_exceeded else "normal"

        if confirmed_crowded:
            occupancy_alerts.inc()
            mode_info = f"auto-zone limit: {effective_limit}" if AUTO_ZONE_MODE else f"manual limit: {effective_limit}"
            alert_msg = (
                f"OCCUPANCY LIMIT EXCEEDED: {persons_detected} people detected "
                f"({mode_info}) on camera '{camera_id or 'default'}'"
            )
            logger.warning(alert_msg)
            _send_alert(alert_msg)
        elif AUTO_ZONE_MODE and zone_capacity is not None:
            logger.info(
                "AutoZone cam='%s': %d/%d people [method=%s]",
                camera_id or "default", persons_detected, zone_capacity, zone_method,
            )

        elapsed = time.time() - start
        detection_latency.observe(elapsed)

        return DetectResponse(
            success=True,
            persons_detected=persons_detected,
            limit_exceeded=limit_exceeded,
            confirmed_crowded=confirmed_crowded,
            status=status_str,
            max_confidence=max_confidence,
            detections=[BoundingBox(**d) for d in detections],
            camera_id=camera_id,
            frame_id=frame_id,
            processing_time_ms=elapsed * 1000,
            auto_zone_mode=AUTO_ZONE_MODE,
            zone_capacity=zone_capacity,
            auto_zone_capacity=zone_capacity if AUTO_ZONE_MODE else None,
            zone_method=zone_method,
        )
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    try:
        import uvloop
        uvloop.install()
    except ImportError:
        pass
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1, loop="asyncio")
