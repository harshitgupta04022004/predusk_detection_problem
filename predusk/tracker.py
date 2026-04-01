from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import cv2

from config import (
    CONF_THRESHOLD,
    IOU_THRESHOLD,
    TRACKER_CONFIG,
    TRACK_BUFFER,
)


@dataclass
class TrackedObject:
    track_id: int
    bbox_xyxy: list[float]
    confidence: float
    class_id: int = 0
    class_name: str = "person"


@dataclass
class TrackingResult:
    frame_index: int
    objects: list[TrackedObject] = field(default_factory=list)

    @property
    def track_ids(self) -> list[int]:
        return [o.track_id for o in self.objects]

    @property
    def count(self) -> int:
        return len(self.objects)


class SportsTracker:
    def __init__(self, weights_path: Path, device: str = ""):
        from ultralytics import YOLO
        self.model = YOLO(str(weights_path))
        self.device = device
        self._reset_state()

    def _reset_state(self):
        self.track_history: dict[int, list] = defaultdict(list)
        self.all_seen_ids: set[int] = set()

    def reset(self):
        self._reset_state()

    def track_frame(self, frame: np.ndarray, frame_index: int) -> TrackingResult:
        results = self.model.track(
            frame,
            persist=True,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            tracker=TRACKER_CONFIG,
            device=self.device,
            verbose=False,
        )

        tracked_objects = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            if boxes.id is not None:
                ids = boxes.id.int().cpu().tolist()
                xyxys = boxes.xyxy.cpu().tolist()
                confs = boxes.conf.cpu().tolist()
                clss = boxes.cls.int().cpu().tolist()
                names = self.model.names

                for tid, xyxy, conf, cls in zip(ids, xyxys, confs, clss):
                    obj = TrackedObject(
                        track_id=tid,
                        bbox_xyxy=xyxy,
                        confidence=conf,
                        class_id=cls,
                        class_name=names.get(cls, "object"),
                    )
                    tracked_objects.append(obj)
                    cx = int((xyxy[0] + xyxy[2]) / 2)
                    cy = int((xyxy[1] + xyxy[3]) / 2)
                    self.track_history[tid].append((cx, cy))
                    if len(self.track_history[tid]) > 60:
                        self.track_history[tid].pop(0)
                    self.all_seen_ids.add(tid)

        return TrackingResult(frame_index=frame_index, objects=tracked_objects)

    def get_trajectory(self, track_id: int) -> list[tuple[int, int]]:
        return self.track_history.get(track_id, [])

    def total_unique_ids(self) -> int:
        return len(self.all_seen_ids)
