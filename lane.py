# dev/creator=tubakhxn
import sys
import subprocess
import importlib


def _ensure(import_name, install_name=None):
    try:
        return importlib.import_module(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", install_name or import_name])
        return importlib.import_module(import_name)


for _imp, _inst in [
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("PIL", "pillow"),
    ("matplotlib", "matplotlib"),
    ("torch", "torch"),
    ("ultralytics", "ultralytics"),
    ("tqdm", "tqdm"),
]:
    _ensure(_imp, _inst)

import os
import time
import argparse
import platform
import traceback
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np
from scipy import interpolate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


@dataclass
class NavConfig:
    src_top_ratio: float = 0.62
    src_top_width: float = 0.42
    src_bottom_width: float = 0.92
    xm_per_pix: float = 3.7 / 700.0
    ym_per_pix: float = 30.0 / 720.0
    n_windows: int = 9
    margin: int = 80
    minpix: int = 40
    smoothing: int = 7
    departure_threshold_m: float = 0.55
    scene_model_name: str = "yolov8n.pt"
    scene_infer_every: int = 5
    output_dir: str = "outputs"
    screenshot_dir: str = "screenshots"
    analytics_dir: str = "analytics"


class PerspectiveTransformer:
    def __init__(self, width, height, cfg: NavConfig):
        self.width = width
        self.height = height
        top_y = int(height * cfg.src_top_ratio)
        top_half = int(width * cfg.src_top_width / 2)
        bot_half = int(width * cfg.src_bottom_width / 2)
        cx = width // 2
        self.src = np.float32([
            [cx - top_half, top_y],
            [cx + top_half, top_y],
            [cx + bot_half, height],
            [cx - bot_half, height],
        ])
        self.dst = np.float32([
            [width * 0.25, 0],
            [width * 0.75, 0],
            [width * 0.75, height],
            [width * 0.25, height],
        ])
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.Minv = cv2.getPerspectiveTransform(self.dst, self.src)

    def warp(self, img):
        return cv2.warpPerspective(img, self.M, (self.width, self.height), flags=cv2.INTER_LINEAR)

    def unwarp(self, img):
        return cv2.warpPerspective(img, self.Minv, (self.width, self.height), flags=cv2.INTER_LINEAR)


class LaneBinarizer:
    def process(self, frame):
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        h, l, s = cv2.split(hls)
        lower_white = np.array([0, 190, 0])
        upper_white = np.array([255, 255, 255])
        white_mask = cv2.inRange(hls, lower_white, upper_white)
        lower_yellow = np.array([15, 30, 90])
        upper_yellow = np.array([35, 200, 255])
        yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)
        color_mask = cv2.bitwise_or(white_mask, yellow_mask)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobelx = np.absolute(sobelx)
        scaled = np.uint8(255 * abs_sobelx / (np.max(abs_sobelx) + 1e-6))
        grad_mask = cv2.inRange(scaled, 30, 255)
        combined = cv2.bitwise_or(color_mask, grad_mask)
        kernel = np.ones((3, 3), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        return combined


class LaneFit:
    def __init__(self, cfg: NavConfig):
        self.cfg = cfg
        self.left_history = deque(maxlen=cfg.smoothing)
        self.right_history = deque(maxlen=cfg.smoothing)

    def sliding_window(self, binary_warped):
        h, w = binary_warped.shape
        histogram = np.sum(binary_warped[h // 2:, :], axis=0)
        midpoint = w // 2
        leftx_base = np.argmax(histogram[:midpoint]) if np.max(histogram[:midpoint]) > 0 else midpoint // 2
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint if np.max(histogram[midpoint:]) > 0 else midpoint + midpoint // 2

        n_windows = self.cfg.n_windows
        window_height = h // n_windows
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        leftx_current = leftx_base
        rightx_current = rightx_base
        margin = self.cfg.margin
        minpix = self.cfg.minpix
        left_lane_inds = []
        right_lane_inds = []

        for window in range(n_windows):
            win_y_low = h - (window + 1) * window_height
            win_y_high = h - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            good_left = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left)
            right_lane_inds.append(good_right)

            if len(good_left) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left]))
            if len(good_right) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right]))

        left_lane_inds = np.concatenate(left_lane_inds) if left_lane_inds else np.array([])
        right_lane_inds = np.concatenate(right_lane_inds) if right_lane_inds else np.array([])

        leftx = nonzerox[left_lane_inds] if len(left_lane_inds) else np.array([])
        lefty = nonzeroy[left_lane_inds] if len(left_lane_inds) else np.array([])
        rightx = nonzerox[right_lane_inds] if len(right_lane_inds) else np.array([])
        righty = nonzeroy[right_lane_inds] if len(right_lane_inds) else np.array([])

        left_fit = np.polyfit(lefty, leftx, 2) if len(lefty) > 300 else None
        right_fit = np.polyfit(righty, rightx, 2) if len(righty) > 300 else None

        if left_fit is not None:
            self.left_history.append(left_fit)
        if right_fit is not None:
            self.right_history.append(right_fit)

        smoothed_left = np.mean(self.left_history, axis=0) if self.left_history else None
        smoothed_right = np.mean(self.right_history, axis=0) if self.right_history else None

        confidence = min(1.0, (len(leftx) + len(rightx)) / (h * 0.6))
        return smoothed_left, smoothed_right, confidence


class NavigationPlanner:
    def __init__(self, cfg: NavConfig, width, height):
        self.cfg = cfg
        self.width = width
        self.height = height

    def curvature_radius(self, fit, y_eval):
        cfg = self.cfg
        if fit is None:
            return 0.0
        A = fit[0] * cfg.xm_per_pix / (cfg.ym_per_pix ** 2)
        B = fit[1] * cfg.xm_per_pix / cfg.ym_per_pix
        curverad = ((1 + (2 * A * y_eval * cfg.ym_per_pix + B) ** 2) ** 1.5) / np.absolute(2 * A + 1e-9)
        return curverad

    def evaluate(self, left_fit, right_fit):
        h, w = self.height, self.width
        y_eval = h - 1
        ploty = np.linspace(0, h - 1, h)

        left_x = None
        right_x = None
        if left_fit is not None:
            left_x = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        if right_fit is not None:
            right_x = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        lane_width_px = None
        center_x = w / 2.0
        vehicle_offset_m = 0.0
        centerline_x = None

        if left_x is not None and right_x is not None:
            lane_width_px = float(np.mean(right_x - left_x))
            centerline_x = (left_x + right_x) / 2.0
            lane_center = centerline_x[-1]
            vehicle_offset_m = (center_x - lane_center) * self.cfg.xm_per_pix
        elif left_x is not None:
            centerline_x = left_x + 200
        elif right_x is not None:
            centerline_x = right_x - 200

        left_rad = self.curvature_radius(left_fit, y_eval)
        right_rad = self.curvature_radius(right_fit, y_eval)
        valid_rads = [r for r in [left_rad, right_rad] if r > 0]
        curvature = float(np.mean(valid_rads)) if valid_rads else 0.0

        steering_angle_deg = 0.0
        if centerline_x is not None:
            dx = centerline_x[max(0, h - 40)] - centerline_x[-1]
            dy = 40.0
            steering_angle_deg = float(np.degrees(np.arctan2(dx, dy)))

        departure = abs(vehicle_offset_m) > self.cfg.departure_threshold_m

        return {
            "ploty": ploty,
            "left_x": left_x,
            "right_x": right_x,
            "centerline_x": centerline_x,
            "lane_width_px": lane_width_px,
            "vehicle_offset_m": vehicle_offset_m,
            "curvature_m": curvature,
            "steering_angle_deg": steering_angle_deg,
            "departure": departure,
        }


class SceneUnderstanding:
    def __init__(self, cfg: NavConfig):
        self.enabled = YOLO is not None
        self.model = None
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.enabled:
            try:
                self.model = YOLO(cfg.scene_model_name)
            except Exception:
                self.enabled = False
        self.last_boxes = []

    def infer(self, frame, frame_idx):
        if not self.enabled or self.model is None:
            return []
        if frame_idx % self.cfg.scene_infer_every != 0:
            return self.last_boxes
        try:
            results = self.model.predict(frame, verbose=False, device=self.device, imgsz=480, conf=0.35)
            boxes = []
            names = results[0].names
            for b in results[0].boxes:
                cls_id = int(b.cls[0])
                label = names.get(cls_id, str(cls_id))
                if label in ("car", "truck", "bus", "motorcycle", "bicycle", "person"):
                    xyxy = b.xyxy[0].cpu().numpy().astype(int)
                    boxes.append((xyxy, label, float(b.conf[0])))
            self.last_boxes = boxes
            return boxes
        except Exception:
            return self.last_boxes


class AnalyticsTracker:
    def __init__(self, maxlen=300):
        self.curvature_history = deque(maxlen=maxlen)
        self.steering_history = deque(maxlen=maxlen)
        self.confidence_history = deque(maxlen=maxlen)
        self.offset_history = deque(maxlen=maxlen)
        self.lane_width_history = deque(maxlen=maxlen)
        self.departure_events = 0
        self._prev_departure = False

    def update(self, nav_result, confidence):
        self.curvature_history.append(min(nav_result["curvature_m"], 5000))
        self.steering_history.append(nav_result["steering_angle_deg"])
        self.confidence_history.append(confidence)
        self.offset_history.append(nav_result["vehicle_offset_m"])
        if nav_result["lane_width_px"]:
            self.lane_width_history.append(nav_result["lane_width_px"])
        if nav_result["departure"] and not self._prev_departure:
            self.departure_events += 1
        self._prev_departure = nav_result["departure"]

    def save_dashboard(self, path):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor="#0d1117")
        fig.suptitle("Vision-Based Lane Navigation Analytics", color="white", fontsize=16)
        for ax in axes.flat:
            ax.set_facecolor("#0d1117")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_color("#30363d")

        axes[0, 0].plot(list(self.curvature_history), color="#00e0ff")
        axes[0, 0].set_title("Road Curvature Radius (m)", color="white")

        axes[0, 1].plot(list(self.steering_history), color="#ff6b6b")
        axes[0, 1].axhline(0, color="gray", linewidth=0.5)
        axes[0, 1].set_title("Steering Angle History (deg)", color="white")

        axes[1, 0].plot(list(self.confidence_history), color="#7cff6b")
        axes[1, 0].set_ylim(0, 1.05)
        axes[1, 0].set_title("Navigation Confidence", color="white")

        axes[1, 1].plot(list(self.offset_history), color="#ffd166")
        axes[1, 1].axhline(0, color="gray", linewidth=0.5)
        axes[1, 1].set_title(f"Lane Offset (m) | Departures: {self.departure_events}", color="white")

        plt.tight_layout()
        fig.savefig(path, facecolor=fig.get_facecolor())
        plt.close(fig)


class HUDRenderer:
    def __init__(self, cfg: NavConfig, width, height):
        self.cfg = cfg
        self.width = width
        self.height = height
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_lane_overlay(self, frame, transformer, nav_result):
        overlay = np.zeros_like(frame)
        ploty = nav_result["ploty"]
        left_x = nav_result["left_x"]
        right_x = nav_result["right_x"]

        if left_x is not None and right_x is not None:
            pts_left = np.array([np.transpose(np.vstack([left_x, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_x, ploty])))])
            pts = np.hstack((pts_left, pts_right))
            cv2.fillPoly(overlay, np.int_([pts]), (40, 180, 60))
            cv2.polylines(overlay, np.int_([pts_left]), False, (0, 255, 255), 8)
            cv2.polylines(overlay, np.int_([pts_right]), False, (0, 255, 255), 8)

        if nav_result["centerline_x"] is not None:
            pts_center = np.array([np.transpose(np.vstack([nav_result["centerline_x"], ploty]))])
            color = (0, 0, 255) if nav_result["departure"] else (255, 255, 255)
            cv2.polylines(overlay, np.int_([pts_center]), False, color, 4)

        unwarped = transformer.unwarp(overlay)
        return cv2.addWeighted(frame, 1.0, unwarped, 0.45, 0)

    def draw_steering_arrow(self, frame, steering_angle_deg):
        h, w = frame.shape[:2]
        cx, cy = w // 2, h - 60
        length = 90
        angle_rad = np.radians(steering_angle_deg)
        ex = int(cx + length * np.sin(angle_rad))
        ey = int(cy - length * np.cos(angle_rad))
        cv2.arrowedLine(frame, (cx, cy), (ex, ey), (0, 220, 255), 4, tipLength=0.35)
        cv2.circle(frame, (cx, cy), 6, (0, 220, 255), -1)
        return frame

    def draw_confidence_meter(self, frame, confidence):
        h, w = frame.shape[:2]
        x0, y0 = 30, h - 40
        bar_w = 220
        cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + 18), (60, 60, 60), -1)
        fill_w = int(bar_w * confidence)
        color = (0, 220, 0) if confidence > 0.6 else ((0, 200, 255) if confidence > 0.3 else (0, 0, 255))
        cv2.rectangle(frame, (x0, y0), (x0 + fill_w, y0 + 18), color, -1)
        cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + 18), (200, 200, 200), 1)
        cv2.putText(frame, f"NAV CONFIDENCE {confidence*100:.0f}%", (x0, y0 - 8), self.font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    def draw_mini_graph(self, frame, history, origin, size, color, label):
        x0, y0 = origin
        w, h = size
        cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (25, 25, 25), -1)
        cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (90, 90, 90), 1)
        data = list(history)
        if len(data) >= 2:
            mn, mx = min(data), max(data)
            rng = (mx - mn) or 1.0
            pts = []
            for i, v in enumerate(data):
                px = x0 + int(i / (len(data) - 1) * w)
                py = y0 + h - int((v - mn) / rng * h)
                pts.append((px, py))
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], color, 2)
        cv2.putText(frame, label, (x0 + 4, y0 + 14), self.font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    def draw_scene_boxes(self, frame, boxes):
        palette = {
            "car": (255, 180, 0),
            "truck": (255, 120, 0),
            "bus": (255, 90, 0),
            "motorcycle": (0, 200, 255),
            "bicycle": (0, 255, 180),
            "person": (0, 0, 255),
        }
        for xyxy, label, conf in boxes:
            x1, y1, x2, y2 = xyxy
            color = palette.get(label, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(15, y1 - 6)), self.font, 0.45, color, 1, cv2.LINE_AA)
        return frame

    def draw_top_panel(self, frame, nav_result, fps, frame_idx):
        h, w = frame.shape[:2]
        panel = frame.copy()
        cv2.rectangle(panel, (0, 0), (w, 90), (10, 10, 10), -1)
        frame = cv2.addWeighted(panel, 0.55, frame, 0.45, 0)

        cv2.putText(frame, "VISION-BASED LANE-LEVEL NAVIGATION", (20, 28), self.font, 0.7, (0, 220, 255), 2, cv2.LINE_AA)
        curvature_text = f"{nav_result['curvature_m']:.0f} m" if nav_result["curvature_m"] > 0 else "N/A"
        cv2.putText(frame, f"Curvature: {curvature_text}", (20, 55), self.font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Steering: {nav_result['steering_angle_deg']:.1f} deg", (20, 78), self.font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        lane_width_m = None
        if nav_result["lane_width_px"]:
            lane_width_m = nav_result["lane_width_px"] * NavConfig().xm_per_pix
        lw_text = f"{lane_width_m:.2f} m" if lane_width_m else "N/A"
        cv2.putText(frame, f"Lane Width: {lw_text}", (340, 55), self.font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Offset: {nav_result['vehicle_offset_m']:.2f} m", (340, 78), self.font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, ts, (w - 230, 28), self.font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 230, 52), self.font, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Frame: {frame_idx}", (w - 230, 74), self.font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        if nav_result["departure"]:
            blink = int(time.time() * 4) % 2 == 0
            if blink:
                cv2.putText(frame, "LANE DEPARTURE WARNING", (w // 2 - 190, 120), self.font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        return frame

    def watermark(self, frame):
        h, w = frame.shape[:2]
        cv2.putText(frame, "tubakhxn", (w - 130, h - 12), self.font, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
        return frame


class VideoProcessor:
    def __init__(self, source, output_path, cfg: NavConfig):
        self.source = source
        self.output_path = output_path
        self.cfg = cfg
        os.makedirs(cfg.output_dir, exist_ok=True)
        os.makedirs(cfg.screenshot_dir, exist_ok=True)
        os.makedirs(cfg.analytics_dir, exist_ok=True)

        self.cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        self.src_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.transformer = PerspectiveTransformer(self.width, self.height, cfg)
        self.binarizer = LaneBinarizer()
        self.lane_fit = LaneFit(cfg)
        self.planner = NavigationPlanner(cfg, self.width, self.height)
        self.scene = SceneUnderstanding(cfg)
        self.analytics = AnalyticsTracker()
        self.hud = HUDRenderer(cfg, self.width, self.height)

        self.writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(output_path, fourcc, self.src_fps, (self.width, self.height))
            if not self.writer.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                self.writer = cv2.VideoWriter(output_path.replace(".mp4", ".avi"), fourcc, self.src_fps, (self.width, self.height))

        self.frame_idx = 0
        self.fps_history = deque(maxlen=30)

    def process_frame(self, frame):
        binary = self.binarizer.process(frame)
        warped = self.transformer.warp(binary)
        left_fit, right_fit, confidence = self.lane_fit.sliding_window(warped)
        nav_result = self.planner.evaluate(left_fit, right_fit)

        output = self.hud.draw_lane_overlay(frame, self.transformer, nav_result)
        boxes = self.scene.infer(frame, self.frame_idx)
        output = self.hud.draw_scene_boxes(output, boxes)
        output = self.hud.draw_steering_arrow(output, nav_result["steering_angle_deg"])
        output = self.hud.draw_confidence_meter(output, confidence)
        output = self.hud.draw_mini_graph(output, self.analytics.curvature_history, (self.width - 260, 100), (230, 80), (0, 224, 255), "CURVATURE")
        output = self.hud.draw_mini_graph(output, self.analytics.steering_history, (self.width - 260, 190), (230, 80), (255, 107, 107), "STEERING")

        fps = 1.0 / (sum(self.fps_history) / len(self.fps_history)) if self.fps_history else 0.0
        output = self.hud.draw_top_panel(output, nav_result, fps, self.frame_idx)
        output = self.hud.watermark(output)

        self.analytics.update(nav_result, confidence)
        return output

    def run(self, show=True):
        progress = tqdm(total=self.total_frames if self.total_frames > 0 else None, desc="Processing", unit="frame")
        try:
            while True:
                t0 = time.time()
                ok, frame = self.cap.read()
                if not ok:
                    break
                frame = cv2.resize(frame, (self.width, self.height))

                try:
                    output = self.process_frame(frame)
                except Exception:
                    traceback.print_exc()
                    output = frame

                if self.writer:
                    self.writer.write(output)

                if show:
                    cv2.imshow("Vision-Based Lane-Level Navigation | tubakhxn", output)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("s"):
                        fname = os.path.join(self.cfg.screenshot_dir, f"lane_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                        cv2.imwrite(fname, output)

                self.frame_idx += 1
                self.fps_history.append(max(time.time() - t0, 1e-6))
                progress.update(1)
        finally:
            progress.close()
            self.cleanup()

    def cleanup(self):
        self.cap.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()
        dashboard_path = os.path.join(self.cfg.analytics_dir, f"lane_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        try:
            self.analytics.save_dashboard(dashboard_path)
            print(f"Analytics dashboard saved to {dashboard_path}")
        except Exception:
            pass


def parse_args():
    parser = argparse.ArgumentParser(description="Vision-Based Lane-Level Navigation | tubakhxn")
    parser.add_argument("--source", type=str, default="0", help="Webcam index or path to video file")
    parser.add_argument("--output", type=str, default="outputs/lane_navigation_output.mp4", help="Path for exported processed video")
    parser.add_argument("--no-show", action="store_true", help="Disable live preview window")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print(" VISION-BASED LANE-LEVEL NAVIGATION | dev/creator=tubakhxn")
    print("=" * 60)
    print(f"Platform: {platform.system()} | CUDA available: {torch.cuda.is_available()}")

    cfg = NavConfig()
    try:
        processor = VideoProcessor(args.source, args.output, cfg)
        processor.run(show=not args.no_show)
    except Exception as exc:
        print(f"Fatal error: {exc}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()