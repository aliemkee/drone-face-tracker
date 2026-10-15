import cv2
import numpy as np
import time

C_GREEN   = (0,   255,  80)
C_RED     = (0,    40, 220)
C_YELLOW  = (0,   220, 240)
C_CYAN    = (220, 200,   0)
C_WHITE   = (240, 240, 240)
C_GRAY    = (120, 120, 120)
C_ORANGE  = (0,   160, 255)
C_BLACK   = (0,     0,   0)
C_MAGENTA = (200,   0, 200)
C_DARK    = (12,   12,  20)

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_MONO = cv2.FONT_HERSHEY_PLAIN


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class Visualizer:
    SIM_W = 340
    SIM_H = 300

    def __init__(self, frame_w: int, frame_h: int):
        self.fw = frame_w
        self.fh = frame_h
        self._fps_buf   = []
        self._last_tick = time.monotonic()
        self._trail     = []

    def draw(self, frame, bbox, state, pid_out, err, sim_state=None):
        out = frame.copy()
        self._draw_scanlines(out)
        self._draw_crosshair(out)
        if bbox is not None:
            self._draw_bbox(out, bbox, state)
            self._draw_vectors(out, err)
        self._draw_status_bar(out, state, pid_out, err)
        self._draw_fps(out)
        if sim_state is not None:
            self._draw_sim_panel(out, sim_state, state, pid_out)
        return out

    def _draw_scanlines(self, img):
        overlay = img.copy()
        for y in range(0, self.fh, 4):
            cv2.line(overlay, (0, y), (self.fw, y), (0, 0, 0), 1)
        cv2.addWeighted(overlay, 0.04, img, 0.96, 0, img)

    def _draw_crosshair(self, img):
        cx, cy = self.fw // 2, self.fh // 2
        cv2.line(img, (cx - 20, cy), (cx - 6, cy),  C_GRAY, 1, cv2.LINE_AA)
        cv2.line(img, (cx + 6,  cy), (cx + 20, cy), C_GRAY, 1, cv2.LINE_AA)
        cv2.line(img, (cx, cy - 20), (cx, cy - 6),  C_GRAY, 1, cv2.LINE_AA)
        cv2.line(img, (cx, cy + 6),  (cx, cy + 20), C_GRAY, 1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 2, C_GRAY, -1, cv2.LINE_AA)

    def _draw_bbox(self, img, bbox, state):
        x, y, w, h = [int(v) for v in bbox]
        x2, y2 = x + w, y + h
        cx, cy = x + w // 2, y + h // 2

        colour = {
            "DETECTING": C_YELLOW,
            "TRACKING":  C_GREEN,
            "LOCKED":    C_ORANGE,
            "HIT":       C_RED,
            "SEARCHING": C_CYAN,
        }.get(state, C_WHITE)

        cl = max(14, min(w, h) // 4)
        lw = 2

        for (ox, oy) in [(x, y), (x2, y), (x, y2), (x2, y2)]:
            dx = 1 if ox == x else -1
            dy = 1 if oy == y else -1
            cv2.line(img, (ox, oy), (ox + dx * cl, oy), colour, lw, cv2.LINE_AA)
            cv2.line(img, (ox, oy), (ox, oy + dy * cl), colour, lw, cv2.LINE_AA)

        cv2.circle(img, (cx, cy), 4, colour, -1, cv2.LINE_AA)
        fc = (self.fw // 2, self.fh // 2)
        cv2.line(img, fc, (cx, cy), colour, 1, cv2.LINE_AA)

        label = state
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.42, 1)
        cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y), colour, -1)
        cv2.putText(img, label, (x + 3, y - 4), FONT, 0.42, C_BLACK, 1, cv2.LINE_AA)

    def _draw_vectors(self, img, err):
        fc = (self.fw // 2, self.fh // 2)
        ex = int(err.get("ex", 0))
        ey = int(err.get("ey", 0))
        if abs(ex) > 8:
            tip = (fc[0] + _clamp(ex // 3, -90, 90), fc[1])
            cv2.arrowedLine(img, fc, tip, C_CYAN, 1, cv2.LINE_AA, tipLength=0.35)
        if abs(ey) > 8:
            tip = (fc[0], fc[1] + _clamp(ey // 3, -90, 90))
            cv2.arrowedLine(img, fc, tip, C_MAGENTA, 1, cv2.LINE_AA, tipLength=0.35)

    def _draw_status_bar(self, img, state, pid_out, err):
        h = 30
        ov = img.copy()
        cv2.rectangle(ov, (0, self.fh - h), (self.fw, self.fh), (10, 10, 15), -1)
        cv2.addWeighted(ov, 0.75, img, 0.25, 0, img)
        colour = {
            "DETECTING": C_YELLOW, "TRACKING": C_GREEN,
            "LOCKED": C_ORANGE,    "HIT": C_RED,
            "SEARCHING": C_CYAN,
        }.get(state, C_WHITE)
        vx  = pid_out.get("vx", 0.0)
        vz  = pid_out.get("vz", 0.0)
        yr  = pid_out.get("yaw_rate", 0.0)
        ea  = err.get("ea", 0.0)
        txt = f"  {state:<11} Vx:{vx:+.2f}  Vz:{vz:+.2f}  Yaw:{yr:+.1f}  dA:{ea:+.3f}"
        cv2.putText(img, txt, (4, self.fh - 9), FONT_MONO, 1.0, colour, 1, cv2.LINE_AA)

    def _draw_fps(self, img):
        now = time.monotonic()
        dt  = now - self._last_tick
        self._last_tick = now
        self._fps_buf.append(1.0 / max(dt, 1e-4))
        if len(self._fps_buf) > 30:
            self._fps_buf.pop(0)
        fps = sum(self._fps_buf) / len(self._fps_buf)
        cv2.putText(img, f"{fps:.0f}fps", (self.fw - 70, 18),
                    FONT, 0.45, C_GRAY, 1, cv2.LINE_AA)

    def _draw_sim_panel(self, img, sim_state, state, pid_out):
        W, H = self.SIM_W, self.SIM_H
        ox   = self.fw - W - 6
        oy   = 6

        panel = np.full((H, W, 3), 10, dtype=np.uint8)

        title_h = 22
        cv2.rectangle(panel, (0, 0), (W, title_h), (20, 20, 35), -1)
        cv2.putText(panel, "SIMULATION", (8, 15), FONT, 0.48, C_CYAN, 1, cv2.LINE_AA)

        colour = {
            "DETECTING": C_YELLOW, "TRACKING": C_GREEN,
            "LOCKED": C_ORANGE,    "HIT": C_RED,
            "SEARCHING": C_CYAN,
        }.get(state, C_WHITE)
        cv2.putText(panel, state, (W - 100, 15), FONT, 0.48, colour, 1, cv2.LINE_AA)

        MAP_Y  = title_h + 4
        MAP_W  = W - 8
        MAP_H  = H - title_h - 80
        SCALE  = 14.0
        map_ox = 4
        map_oy = MAP_Y

        cv2.rectangle(panel, (map_ox, map_oy), (map_ox + MAP_W, map_oy + MAP_H),
                      (25, 25, 40), -1)

        grid_step = int(SCALE)
        mcx = map_ox + MAP_W // 2
        mcy = map_oy + MAP_H // 2
        for gx in range(mcx % grid_step, map_ox + MAP_W, grid_step):
            cv2.line(panel, (gx, map_oy), (gx, map_oy + MAP_H), (30, 30, 45), 1)
        for gy in range(mcy % grid_step, map_oy + MAP_H, grid_step):
            cv2.line(panel, (map_ox, gy), (map_ox + MAP_W, gy), (30, 30, 45), 1)

        cv2.line(panel, (mcx, map_oy), (mcx, map_oy + MAP_H), (40, 40, 60), 1)
        cv2.line(panel, (map_ox, mcy), (map_ox + MAP_W, mcy), (40, 40, 60), 1)
        cv2.circle(panel, (mcx, mcy), 3, (60, 60, 80), -1)

        dpx = _clamp(mcx + int(sim_state.x * SCALE), map_ox + 4, map_ox + MAP_W - 4)
        dpy = _clamp(mcy - int(sim_state.y * SCALE), map_oy + 4, map_oy + MAP_H - 4)

        self._trail.append((dpx, dpy))
        if len(self._trail) > 120:
            self._trail.pop(0)

        for i in range(1, len(self._trail)):
            alpha = i / len(self._trail)
            tc = tuple(int(c * alpha) for c in C_CYAN)
            cv2.line(panel, self._trail[i - 1], self._trail[i], tc, 1)

        yaw_r  = float(np.radians(sim_state.yaw))
        arr_len = 12
        arr_tip = (
            int(dpx + arr_len * np.cos(yaw_r)),
            int(dpy - arr_len * np.sin(yaw_r)),
        )
        cv2.arrowedLine(panel, (dpx, dpy), arr_tip, C_GREEN, 2, cv2.LINE_AA, tipLength=0.4)
        cv2.circle(panel, (dpx, dpy), 6, C_CYAN,  -1, cv2.LINE_AA)
        cv2.circle(panel, (dpx, dpy), 8, C_WHITE,  1, cv2.LINE_AA)

        info_y = map_oy + MAP_H + 10
        lines = [
            (f"X:{sim_state.x:+6.2f}m  Y:{sim_state.y:+6.2f}m", C_WHITE),
            (f"Z:{sim_state.z:+6.2f}m  Yaw:{sim_state.yaw % 360:.1f}", C_WHITE),
            (f"Vx:{pid_out.get('vx', 0):+5.2f}  Vz:{pid_out.get('vz', 0):+5.2f}  "
             f"Yr:{pid_out.get('yaw_rate', 0):+5.1f}", C_CYAN),
        ]
        for i, (txt, col) in enumerate(lines):
            cv2.putText(panel, txt, (6, info_y + i * 18),
                        FONT_MONO, 0.95, col, 1, cv2.LINE_AA)

        cv2.rectangle(panel, (0, 0), (W - 1, H - 1), (50, 50, 70), 1)

        roi = img[oy:oy + H, ox:ox + W]
        cv2.addWeighted(panel, 0.88, roi, 0.12, 0, roi)
        img[oy:oy + H, ox:ox + W] = roi
