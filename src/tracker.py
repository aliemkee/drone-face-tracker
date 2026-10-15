import cv2
import numpy as np
from enum import Enum, auto


class TrackerState(Enum):
    IDLE     = auto()
    TRACKING = auto()
    LOST     = auto()


class CSRTTracker:
    MAX_LOST_FRAMES  = 6
    MIN_BOX_AREA     = 400
    EDGE_MARGIN      = 8
    MAX_SCALE_CHANGE = 2.8

    def __init__(self):
        self._tracker     = None
        self._state       = TrackerState.IDLE
        self._last_bbox   = None
        self._init_bbox   = None
        self._lost_frames = 0

    def init(self, frame: np.ndarray, bbox: tuple):
        fh, fw = frame.shape[:2]
        x, y, w, h = [int(v) for v in bbox]
        x = max(0, min(x, fw - 1))
        y = max(0, min(y, fh - 1))
        w = max(1, min(w, fw - x))
        h = max(1, min(h, fh - y))
        self._tracker = cv2.TrackerCSRT_create()
        ok = self._tracker.init(frame, (x, y, w, h))
        self._init_bbox   = (x, y, w, h)
        self._last_bbox   = (x, y, w, h)
        self._lost_frames = 0
        self._state = TrackerState.TRACKING if ok else TrackerState.LOST
        return ok

    def update(self, frame: np.ndarray):
        if self._tracker is None or self._state == TrackerState.IDLE:
            return False, None

        success, box = self._tracker.update(frame)
        fh, fw = frame.shape[:2]

        if not success:
            return self._mark_lost()

        x, y, w, h = [int(v) for v in box]

        if x + w <= self.EDGE_MARGIN or x >= fw - self.EDGE_MARGIN:
            return self._mark_lost()
        if y + h <= self.EDGE_MARGIN or y >= fh - self.EDGE_MARGIN:
            return self._mark_lost()

        if w * h < self.MIN_BOX_AREA:
            return self._mark_lost()

        if self._init_bbox:
            iw, ih = self._init_bbox[2], self._init_bbox[3]
            ratio = max(w / max(iw, 1), iw / max(w, 1), h / max(ih, 1), ih / max(h, 1))
            if ratio > self.MAX_SCALE_CHANGE:
                return self._mark_lost()

        self._lost_frames = 0
        self._last_bbox   = (x, y, w, h)
        self._state       = TrackerState.TRACKING
        return True, (x, y, w, h)

    def _mark_lost(self):
        self._lost_frames += 1
        if self._lost_frames >= self.MAX_LOST_FRAMES:
            self._state = TrackerState.LOST
        return False, self._last_bbox

    def reset(self):
        self._tracker     = None
        self._state       = TrackerState.IDLE
        self._last_bbox   = None
        self._init_bbox   = None
        self._lost_frames = 0

    @property
    def state(self):
        return self._state

    @property
    def is_tracking(self):
        return self._state == TrackerState.TRACKING

    @property
    def last_bbox(self):
        return self._last_bbox
