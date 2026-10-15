import cv2
import time
import sys
import argparse
from pathlib import Path

SRC = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC))

from detector         import FaceDetector
from tracker          import CSRTTracker, TrackerState
from pid_controller   import PIDController
from drone_controller import DroneController, VelocityCmd
from visualizer       import Visualizer


CFG = {
    "camera_index"   : 0,
    "frame_width"    : 640,
    "frame_height"   : 480,

    "yolo_model"     : None,
    "yolo_conf"      : 0.45,
    "detect_every"   : 12,

    "pid_yaw"        : (0.040, 0.0003, 0.018),
    "pid_pitch"      : (0.030, 0.0002, 0.012),
    "pid_fwd"        : (1.20,  0.005,  0.40),
    "pid_limit"      : 1.0,
    "integral_limit" : 0.4,

    "speed_scale"    : 2.0,
    "yaw_scale"      : 60.0,

    "dz_xy_px"       : 12,
    "dz_area"        : 0.005,

    "target_area"    : 0.10,
    "hit_area"       : 0.35,

    "edge_margin_pct": 0.06,

    "reacquire_frames": 45,

    "mavlink_conn"   : "udp:127.0.0.1:14550",
    "max_speed"      : 2.0,
    "max_yaw"        : 60.0,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--real",    action="store_true")
    p.add_argument("--connect", default=CFG["mavlink_conn"])
    p.add_argument("--cam",     type=int, default=CFG["camera_index"])
    p.add_argument("--haar",    action="store_true")
    return p.parse_args()


def dead_zone(v, th):
    return 0.0 if abs(v) < th else v


def compute_errors(bbox, fw, fh, target_area):
    x, y, w, h = bbox
    err_x  = (x + w / 2) - fw / 2
    err_y  = (y + h / 2) - fh / 2
    area_r = (w * h) / (fw * fh)
    err_a  = area_r - target_area
    return err_x, err_y, err_a, area_r


def is_near_edge(bbox, fw, fh, margin_pct):
    x, y, w, h = bbox
    m = margin_pct
    cx = (x + w / 2) / fw
    cy = (y + h / 2) / fh
    return cx < m or cx > 1 - m or cy < m or cy > 1 - m


def iou(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ix1 = max(ax1, bx1);  iy1 = max(ay1, by1)
    ix2 = min(ax1+aw, bx1+bw);  iy2 = min(ay1+ah, by1+bh)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    union = aw*ah + bw*bh - inter
    return inter / union if union > 0 else 0.0


def best_match(boxes, ref_bbox):
    if not boxes or ref_bbox is None:
        return boxes[0] if boxes else None
    scored = [(iou(b, ref_bbox), b) for b in boxes]
    scored.sort(reverse=True)
    return scored[0][1] if scored[0][0] > 0.15 else boxes[0]


def main():
    args  = parse_args()

    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CFG["frame_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG["frame_height"])
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    if not cap.isOpened():
        sys.exit(f"Cannot open camera {args.cam}")

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = FaceDetector(model_path=CFG["yolo_model"],
                            conf_threshold=CFG["yolo_conf"],
                            use_haar=args.haar)
    tracker  = CSRTTracker()
    viz      = Visualizer(fw, fh)

    pid_yaw   = PIDController(*CFG["pid_yaw"],   output_limit=CFG["pid_limit"], integral_limit=CFG["integral_limit"])
    pid_pitch = PIDController(*CFG["pid_pitch"],  output_limit=CFG["pid_limit"], integral_limit=CFG["integral_limit"])
    pid_fwd   = PIDController(*CFG["pid_fwd"],    output_limit=CFG["pid_limit"], integral_limit=CFG["integral_limit"])

    drone = DroneController(connection_string=args.connect,
                            sim=not args.real,
                            max_speed=CFG["max_speed"],
                            max_yaw_rate=CFG["max_yaw"])
    if not drone.connect():
        sys.exit("Could not connect to drone.")

    cv2.namedWindow("Drone Face Tracker", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Drone Face Tracker", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    state          = "DETECTING"
    frame_count    = 0
    paused         = False
    hit_time       = None
    search_frames  = 0
    last_bbox      = None

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.03)
            continue
        frame_count += 1

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('r'):
            tracker.reset()
            for p in (pid_yaw, pid_pitch, pid_fwd): p.reset()
            state = "DETECTING"
            last_bbox = None
        if key == ord(' '):
            paused = not paused
        if key == ord('s'):
            cv2.imwrite(f"shot_{time.strftime('%H%M%S')}.png", frame)

        if paused:
            cv2.putText(frame, "PAUSED", (fw//2 - 55, fh//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 200, 255), 2)
            cv2.imshow("Drone Face Tracker", frame)
            drone.stop()
            continue

        bbox      = None
        pid_out   = {"vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw_rate": 0.0}
        err_dict  = {"ex": 0.0, "ey": 0.0, "ea": 0.0, "area": 0.0}

        if state in ("DETECTING", "SEARCHING"):
            boxes = detector.detect(frame)
            if boxes:
                bbox  = best_match(boxes, last_bbox)
                tracker.init(frame, bbox)
                last_bbox     = bbox
                search_frames = 0
                state = "TRACKING"
                for p in (pid_yaw, pid_pitch, pid_fwd): p.reset()
            else:
                search_frames += 1
                drone.stop()
                if search_frames > CFG["reacquire_frames"] * 3:
                    state = "DETECTING"
                    last_bbox = None

        else:
            ok, bbox = tracker.update(frame)

            if bbox is not None:
                last_bbox = bbox

            face_left_frame = ok and is_near_edge(bbox, fw, fh, CFG["edge_margin_pct"])

            if face_left_frame:
                tracker.reset()
                for p in (pid_yaw, pid_pitch, pid_fwd): p.reset()
                state         = "SEARCHING"
                search_frames = 0
                bbox          = None
                drone.stop()

            elif not ok:
                boxes = detector.detect(frame)
                if boxes:
                    bbox  = best_match(boxes, last_bbox)
                    tracker.init(frame, bbox)
                    last_bbox = bbox
                    state = "TRACKING"
                    for p in (pid_yaw, pid_pitch, pid_fwd): p.reset()
                else:
                    search_frames += 1
                    state = "SEARCHING" if search_frames < CFG["reacquire_frames"] else "DETECTING"
                    drone.stop()
                    bbox = None

            else:
                search_frames = 0
                if frame_count % CFG["detect_every"] == 0:
                    boxes = detector.detect(frame)
                    if boxes:
                        candidate = best_match(boxes, bbox)
                        if iou(candidate, bbox) > 0.20:
                            bbox = candidate
                            tracker.init(frame, bbox)
                            last_bbox = bbox

        if bbox is not None and state not in ("DETECTING", "SEARCHING"):
            ex, ey, ea, area = compute_errors(bbox, fw, fh, CFG["target_area"])
            err_dict = {"ex": ex, "ey": ey, "ea": ea, "area": area}

            out_yaw   = pid_yaw.compute(dead_zone(ex, CFG["dz_xy_px"]))
            out_pitch = pid_pitch.compute(dead_zone(ey, CFG["dz_xy_px"]))
            out_fwd   = pid_fwd.compute(dead_zone(ea, CFG["dz_area"]))

            vx       = -out_fwd   * CFG["speed_scale"]
            vz       =  out_pitch * CFG["speed_scale"]
            yaw_rate =  out_yaw   * CFG["yaw_scale"]

            pid_out = {"vx": vx, "vy": 0.0, "vz": vz, "yaw_rate": yaw_rate}

            if area >= CFG["hit_area"]:
                state    = "HIT"
                hit_time = time.monotonic()
                drone.stop()
            elif abs(ex) < 25 and abs(ey) < 25 and abs(ea) < 0.02:
                state = "LOCKED"
            else:
                state = "TRACKING"

            if state != "HIT":
                drone.send_velocity(VelocityCmd(**pid_out))

        if state == "HIT" and hit_time and (time.monotonic() - hit_time > 2.0):
            hit_time  = None
            state     = "DETECTING"
            last_bbox = None
            tracker.reset()
            for p in (pid_yaw, pid_pitch, pid_fwd): p.reset()

        display = viz.draw(frame, bbox, state, pid_out, err_dict,
                           drone.sim_state if not args.real else None)
        cv2.imshow("Drone Face Tracker", display)

    drone.land()
    drone.disconnect()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
