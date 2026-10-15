import cv2
import numpy as np
from pathlib import Path

try:
    import onnxruntime as ort
    _ORT_OK = True
except ImportError:
    _ORT_OK = False

_HAAR_PATH  = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_MODEL_PATH = Path(__file__).parent.parent / "models" / "yolov8n-face.onnx"


class FaceDetector:
    def __init__(self, model_path=None, conf_threshold=0.45,
                 use_haar=False, input_size=(640, 640)):
        self.conf   = conf_threshold
        self.insize = input_size
        self._mode  = "none"

        if use_haar or not _ORT_OK:
            self._init_haar()
            return

        mp = Path(model_path) if model_path else _MODEL_PATH
        if mp.exists():
            self._init_onnx(mp)
        else:
            print(f"[Detector] ONNX not found: {mp}  →  run download_model.py")
            self._init_haar()

    def _init_onnx(self, path):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self._inp  = self._sess.get_inputs()[0].name
        self._mode = "onnx"
        print(f"[Detector] ONNX → {path.name}")

    def _init_haar(self):
        self._haar = cv2.CascadeClassifier(_HAAR_PATH)
        if self._haar.empty():
            raise RuntimeError("Haar cascade missing.")
        self._mode = "haar"
        print("[Detector] Haar cascade")

    def detect(self, frame):
        return self._detect_onnx(frame) if self._mode == "onnx" else self._detect_haar(frame)

    def _detect_onnx(self, frame):
        ih, iw = frame.shape[:2]
        tw, th = self.insize
        blob, (px, py, scale) = self._letterbox(frame, tw, th)
        blob = blob.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]
        raw  = self._sess.run(None, {self._inp: blob})[0]
        out  = raw[0]
        if out.shape[0] == 5:
            out = out.T
        elif out.shape[0] != 5 and out.shape[1] != 5:
            out = raw[0].T
            scores = out[:, 4] if out.shape[1] == 5 else out[:, 4:].max(axis=1)
            out    = np.concatenate([out[:, :4], scores[:, None]], axis=1)
        mask = out[:, 4] >= self.conf
        out  = out[mask]
        if not len(out):
            return []
        xyxy   = self._cx2xy(out[:, :4])
        scores = out[:, 4].tolist()
        idxs   = cv2.dnn.NMSBoxes([list(map(float, b)) for b in xyxy], scores, self.conf, 0.45)
        if hasattr(idxs, "flatten"):
            idxs = idxs.flatten().tolist()
        result = []
        for i in idxs:
            x1, y1, x2, y2 = xyxy[i]
            x1 = max(0, int((x1 - px) / scale))
            y1 = max(0, int((y1 - py) / scale))
            x2 = min(iw, int((x2 - px) / scale))
            y2 = min(ih, int((y2 - py) / scale))
            if x2 > x1 and y2 > y1:
                result.append((x1, y1, x2 - x1, y2 - y1))
        result.sort(key=lambda b: b[2] * b[3], reverse=True)
        return result

    def _detect_haar(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.equalizeHist(gray, gray)
        rects = self._haar.detectMultiScale(gray, 1.1, 5, minSize=(60, 60),
                                            flags=cv2.CASCADE_SCALE_IMAGE)
        if not len(rects):
            return []
        boxes = [tuple(r) for r in rects]
        boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
        return boxes

    @staticmethod
    def _letterbox(img, tw, th):
        ih, iw = img.shape[:2]
        s = min(tw / iw, th / ih)
        nw, nh = int(iw * s), int(ih * s)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas  = np.full((th, tw, 3), 114, dtype=np.uint8)
        px, py  = (tw - nw) // 2, (th - nh) // 2
        canvas[py:py+nh, px:px+nw] = resized
        return canvas, (px, py, s)

    @staticmethod
    def _cx2xy(b):
        o = np.empty_like(b)
        o[:, 0] = b[:, 0] - b[:, 2] / 2
        o[:, 1] = b[:, 1] - b[:, 3] / 2
        o[:, 2] = b[:, 0] + b[:, 2] / 2
        o[:, 3] = b[:, 1] + b[:, 3] / 2
        return o

    @property
    def mode(self):
        return self._mode
