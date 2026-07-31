"""Auto-zone capacity estimation for desk/workspace cameras.

When MAX_PEOPLE=0, this module estimates a per-camera occupancy limit from the
current camera view. It uses the best available signal in this order:

1. Chair count: COCO class 56, one chair is one seat.
2. Table area: COCO class 60, table pixels divided by average person area.
3. Spatial density: frame area divided by average detected person area.

Capacity estimates are cached per camera and recalculated every 60 seconds.
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

_CHAIR_CLS = 56
_TABLE_CLS = 60

RECALIBRATE_INTERVAL = float(os.environ.get("ZONE_RECALIBRATE_INTERVAL", "60"))
DESK_SPACING_FACTOR = 1.8
SPATIAL_PERSON_SPACE_FACTOR = 2.5


def _env_bool(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class ZoneCapacityEstimator:
    """Per-camera capacity estimator backed by YOLO COCO detections."""

    def __init__(self, min_capacity: int = 1, max_capacity: int = 100) -> None:
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.use_chairs = _env_bool("AUTO_ZONE_USE_CHAIRS", True)
        self.use_table_area = _env_bool("AUTO_ZONE_USE_TABLE_AREA", True)
        self.use_spatial_density = _env_bool("AUTO_ZONE_USE_SPATIAL_DENSITY", True)
        self._cache: dict[str, dict[str, Any]] = {}

        logger.info(
            "[AutoZone] sources enabled: chairs=%s, table_area=%s, spatial_density=%s",
            self.use_chairs,
            self.use_table_area,
            self.use_spatial_density,
        )

    def estimate_capacity(
        self,
        frame: np.ndarray,
        model: Any,
    ) -> tuple[Optional[int], str]:
        """Compatibility wrapper for callers that do not pass camera metadata."""
        return self.get_capacity("default", frame, model, [])

    def get_capacity(
        self,
        camera_id: str,
        frame: np.ndarray,
        model: Any,
        person_detections: list[dict],
    ) -> tuple[Optional[int], str]:
        """Return (estimated_capacity, method_used) for one camera frame."""
        now = time.monotonic()
        cached = self._cache.get(camera_id)

        if cached and (now - float(cached["last_updated"])) < RECALIBRATE_INTERVAL:
            return int(cached["capacity"]), str(cached["method"])

        capacity = None
        method = ""

        if self.use_chairs or self.use_table_area:
            capacity, method = self._estimate_from_furniture(
                frame,
                model,
                person_detections,
            )

        if capacity is None and self.use_spatial_density:
            capacity = self._estimate_from_spatial_density(frame, person_detections)
            method = "spatial_density" if capacity is not None else "spatial_waiting_for_people"

        if capacity is None:
            return None, method or "no_enabled_source_matched"

        capacity = max(self.min_capacity, min(self.max_capacity, int(capacity)))
        self._cache[camera_id] = {
            "capacity": capacity,
            "method": method,
            "last_updated": now,
        }
        logger.info(
            "[AutoZone] Camera '%s' -> estimated capacity=%d (method=%s)",
            camera_id,
            capacity,
            method,
        )
        return capacity, method

    def cached_capacities(self) -> dict[str, int]:
        """Return the current per-camera capacity cache."""
        return {
            camera_id: int(entry["capacity"])
            for camera_id, entry in self._cache.items()
            if entry.get("capacity") is not None
        }

    def is_calibrating(self, camera_id: str) -> bool:
        """True until this camera has its first cached capacity estimate."""
        return camera_id not in self._cache

    def calibration_progress(self, camera_id: str) -> float:
        """Return 1.0 once a capacity estimate exists, else 0.0."""
        return 0.0 if self.is_calibrating(camera_id) else 1.0

    def _estimate_from_furniture(
        self,
        frame: np.ndarray,
        model: Any,
        person_detections: list[dict],
    ) -> tuple[Optional[int], str]:
        """Estimate capacity from chairs first, then tables."""
        classes = []
        if self.use_chairs:
            classes.append(_CHAIR_CLS)
        if self.use_table_area:
            classes.append(_TABLE_CLS)

        if not classes:
            return None, ""

        try:
            furniture_results = model.predict(
                frame,
                classes=classes,
                conf=0.35,
                verbose=False,
            )
        except Exception as exc:
            logger.warning("[AutoZone] Furniture detection failed: %s", exc)
            return None, ""

        chairs, tables = self._parse_furniture(furniture_results)

        if self.use_chairs and chairs:
            return len(chairs), "chair_count"

        if self.use_table_area and tables:
            avg_person_area = self._avg_person_area(frame, person_detections)
            table_area_px = sum(
                max(0, x2 - x1) * max(0, y2 - y1)
                for (x1, y1, x2, y2) in tables
            )
            if avg_person_area > 0:
                capacity = table_area_px / (avg_person_area * DESK_SPACING_FACTOR)
                return max(1, int(math.floor(capacity))), "table_area"

        return None, ""

    def _estimate_from_spatial_density(
        self,
        frame: np.ndarray,
        person_detections: list[dict],
    ) -> Optional[int]:
        """Estimate capacity from frame area and visible person bbox size."""
        if not person_detections:
            return None

        avg_person_area = self._avg_person_area(frame, person_detections)
        if avg_person_area <= 0:
            return None

        h, w = frame.shape[:2]
        frame_area = float(h * w)
        capacity = frame_area / (avg_person_area * SPATIAL_PERSON_SPACE_FACTOR)
        return max(1, int(math.floor(capacity)))

    @staticmethod
    def _parse_furniture(results: list) -> tuple[list[tuple], list[tuple]]:
        """Parse YOLO results for chair and table bounding boxes."""
        chairs: list[tuple] = []
        tables: list[tuple] = []

        if not results:
            return chairs, tables

        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue

            xyxy = boxes.xyxy.cpu().int().tolist()
            cls_ids = boxes.cls.cpu().int().tolist()

            for (x1, y1, x2, y2), cls_id in zip(xyxy, cls_ids):
                if cls_id == _CHAIR_CLS:
                    chairs.append((x1, y1, x2, y2))
                elif cls_id == _TABLE_CLS:
                    tables.append((x1, y1, x2, y2))

        return chairs, tables

    @staticmethod
    def _avg_person_area(frame: np.ndarray, person_detections: list[dict]) -> float:
        """Return average detected person bounding box area in pixels."""
        areas = []

        for det in person_detections:
            bbox = det.get("bbox")
            if bbox:
                x1, y1, x2, y2 = bbox
            else:
                x1 = det.get("x1", 0)
                y1 = det.get("y1", 0)
                x2 = det.get("x2", 0)
                y2 = det.get("y2", 0)

            area = max(0, (x2 - x1) * (y2 - y1))
            if area > 0:
                areas.append(area)

        if areas:
            return float(sum(areas) / len(areas))

        h, w = frame.shape[:2]
        return float((h / 6) * (w / 10))
