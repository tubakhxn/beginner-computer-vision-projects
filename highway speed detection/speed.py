# dev/creator: tubakhxn

import sys
import subprocess
import importlib

REQUIRED = {
    "cv2": "opencv-python",
    "numpy": "numpy",
    "ultralytics": "ultralytics",
    "tqdm": "tqdm",
    "lap": "lapx",
}

for module_name, pip_name in REQUIRED.items():
    try:
        importlib.import_module(module_name)
    except ImportError:
        print(f"[setup] Installing missing dependency: {pip_name} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pip_name])

import os
import time
import math
from collections import defaultdict, deque

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else "speed.mp4"

PROCESS_WIDTH = 1280
DETECT_IMGSZ = 480
FRAME_SKIP = 2
MODEL_NAME = "yolov8n.pt"
VEHICLE_CLASSES = [2, 3, 5, 7]

SPEED_LIMIT_KMH = 35
SMOOTHING_WINDOW = 8
MASK_ALPHA = 0.45
LABEL_COLOR = (255, 120, 0)

LANE_WIDTH_M = 3.7
BEV_WIDTH, BEV_HEIGHT = 400, 1000
METERS_PER_PIXEL = LANE_WIDTH_M / (BEV_WIDTH / 4)

ROAD_TRAPEZOID_FRACTIONS = [
    (0.35, 0.30),
    (0.75, 0.30),
    (0.00, 0.98),
    (1.00, 0.98),
]

if not os.path.exists(VIDEO_PATH):
    print(f"Error: could not find video file '{VIDEO_PATH}'")
    sys.exit(1)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: could not open video file '{VIDEO_PATH}'")
    sys.exit(1)

orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

scale = PROCESS_WIDTH / orig_width
proc_width = PROCESS_WIDTH
proc_height = int(orig_height * scale)

base_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
output_path = f"{base_name}_tracked.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (proc_width, proc_height))

src_points = np.float32([
    (fx * proc_width, fy * proc_height) for fx, fy in ROAD_TRAPEZOID_FRACTIONS
])
dst_points = np.float32([
    [0, 0],
    [BEV_WIDTH, 0],
    [0, BEV_HEIGHT],
    [BEV_WIDTH, BEV_HEIGHT],
])
matrix = cv2.getPerspectiveTransform(src_points, dst_points)

print("[setup] Loading YOLOv8 model...")
model = YOLO(MODEL_NAME)

try:
    import torch
    device = 0 if torch.cuda.is_available() else "cpu"
except ImportError:
    device = "cpu"
print(f"[setup] Running on device: {device}")

position_history = defaultdict(lambda: deque(maxlen=SMOOTHING_WINDOW))
speed_history = defaultdict(lambda: deque(maxlen=SMOOTHING_WINDOW))
last_seen_speed = defaultdict(float)

session_top_speed = 0.0
session_top_id = None


def fill_vehicle(frame, x1, y1, x2, y2, color, alpha=MASK_ALPHA):
    h, w = frame.shape[:2]
    x1c, y1c = max(x1, 0), max(y1, 0)
    x2c, y2c = min(x2, w - 1), min(y2, h - 1)
    if x2c <= x1c or y2c <= y1c:
        return
    roi = frame[y1c:y2c, x1c:x2c]
    colored = np.full_like(roi, color)
    frame[y1c:y2c, x1c:x2c] = cv2.addWeighted(colored, alpha, roi, 1 - alpha, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


def speed_color(speed_kmh):
    if speed_kmh > SPEED_LIMIT_KMH:
        return (0, 0, 255)
    return (0, 230, 0)


print(f"[run] Processing '{VIDEO_PATH}' ({orig_width}x{orig_height} -> {proc_width}x{proc_height}) ...")

pbar = tqdm(total=total_frames if total_frames > 0 else None, unit="frame", desc="Tracking")
start_time = time.time()
frame_idx = 0

last_drawn = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    frame = cv2.resize(frame, (proc_width, proc_height))
    run_detection = (frame_idx % FRAME_SKIP == 0) or frame_idx == 1

    if run_detection:
        results = model.track(
            frame,
            persist=True,
            classes=VEHICLE_CLASSES,
            tracker="bytetrack.yaml",
            device=device,
            imgsz=DETECT_IMGSZ,
            verbose=False,
        )[0]

        last_drawn = {}

        if results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, y2

                point = np.array([[[cx, cy]]], dtype=np.float32)
                bev_point = cv2.perspectiveTransform(point, matrix)[0][0]

                position_history[track_id].append((bev_point[0], bev_point[1], frame_idx))

                speed_kmh = last_seen_speed[track_id]
                if len(position_history[track_id]) >= 2:
                    (x_old, y_old, f_old) = position_history[track_id][0]
                    (x_new, y_new, f_new) = position_history[track_id][-1]
                    frame_gap = max(f_new - f_old, 1)
                    pixel_dist = math.hypot(x_new - x_old, y_new - y_old)
                    meters = pixel_dist * METERS_PER_PIXEL
                    seconds = frame_gap / fps
                    mps = meters / seconds if seconds > 0 else 0
                    speed_kmh = mps * 3.6

                    speed_history[track_id].append(speed_kmh)
                    speed_kmh = sum(speed_history[track_id]) / len(speed_history[track_id])
                    last_seen_speed[track_id] = speed_kmh

                last_drawn[track_id] = (x1, y1, x2, y2, speed_kmh)

                if speed_kmh > session_top_speed:
                    session_top_speed = speed_kmh
                    session_top_id = track_id

    for tid, (x1, y1, x2, y2, spd) in last_drawn.items():
        if spd < 5:
            continue

        color = speed_color(spd)
        fill_vehicle(frame, x1, y1, x2, y2, color)

        label = f"{spd:.0f} km/h"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 10, y1), LABEL_COLOR, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        if spd > SPEED_LIMIT_KMH:
            cv2.putText(frame, "SPEEDING", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

    hud_h = 110
    hud_roi = frame[20:20 + hud_h, 20:430]
    dark = np.full_like(hud_roi, (15, 15, 15))
    frame[20:20 + hud_h, 20:430] = cv2.addWeighted(dark, 0.75, hud_roi, 0.25, 0)
    cv2.rectangle(frame, (20, 20), (430, 20 + hud_h), (0, 255, 200), 1)

    cv2.putText(frame, "HIGHWAY SPEED TRACKER", (35, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)
    vehicles_now = len(last_drawn)
    cv2.putText(frame, f"Vehicles tracked: {vehicles_now}", (35, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(frame, f"Speed limit: {SPEED_LIMIT_KMH} km/h", (35, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
    cv2.putText(frame, f"Session top speed: {session_top_speed:.0f} km/h", (35, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1)

    out.write(frame)

    elapsed = time.time() - start_time
    live_fps = frame_idx / elapsed if elapsed > 0 else 0
    pbar.set_postfix({"vehicles": vehicles_now, "top_kmh": f"{session_top_speed:.0f}", "fps": f"{live_fps:.1f}"})
    pbar.update(1)

pbar.close()
cap.release()
out.release()

print(f"\n[done] Saved output to: {output_path}")
print(f"[done] Session top speed recorded: {session_top_speed:.1f} km/h (track ID {session_top_id})")