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
from collections import defaultdict, deque

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else "wrong.mp4"

PROCESS_WIDTH = 1280          
DETECT_IMGSZ = 480            
FRAME_SKIP = 2                
MODEL_NAME = "yolov8n.pt"      
VEHICLE_CLASSES = [2, 3, 5, 7] 

MASK_ALPHA = 0.45               
LABEL_COLOR_NORMAL = (0, 150, 0)    
LABEL_COLOR_WRONG = (0, 0, 200)      

HISTORY_LEN = 15                
MIN_HISTORY_FOR_VERDICT = 6    
MIN_DISPLACEMENT_PX = 8        
DOMINANT_FLOW_SMOOTHING = 60   
WRONG_WAY_VOTE_MARGIN = 0.15   


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
output_path = f"{base_name}_wrongway.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (proc_width, proc_height))


print("[setup] Loading YOLOv8 model...")
model = YOLO(MODEL_NAME)

try:
    import torch
    device = 0 if torch.cuda.is_available() else "cpu"
except ImportError:
    device = "cpu"
print(f"[setup] Running on device: {device}")

position_history = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
verdict_history = defaultdict(lambda: deque(maxlen=10))  
wrong_way_ids_seen = set()


direction_votes = deque(maxlen=DOMINANT_FLOW_SMOOTHING)


def track_net_direction(history):
    """Returns net vertical displacement (last.y - first.y) and horizontal
    displacement for a track's recent history. Positive dy = moving down
    the frame (toward camera), negative dy = moving up (away from camera)."""
    if len(history) < 2:
        return 0.0, 0.0
    (x_old, y_old, _), (x_new, y_new, _) = history[0], history[-1]
    return (y_new - y_old), (x_new - x_old)


def fill_vehicle(frame, x1, y1, x2, y2, color, alpha=MASK_ALPHA):
    """Fills a vehicle's box with a translucent color (segmentation-style look).
    Only touches the box region, not the full frame, so it stays fast even
    with many vehicles on screen."""
    h, w = frame.shape[:2]
    x1c, y1c = max(x1, 0), max(y1, 0)
    x2c, y2c = min(x2, w - 1), min(y2, h - 1)
    if x2c <= x1c or y2c <= y1c:
        return
    roi = frame[y1c:y2c, x1c:x2c]
    colored = np.full_like(roi, color)
    frame[y1c:y2c, x1c:x2c] = cv2.addWeighted(colored, alpha, roi, 1 - alpha, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


print(f"[run] Processing '{VIDEO_PATH}' ({orig_width}x{orig_height} -> {proc_width}x{proc_height}) ...")

pbar = tqdm(total=total_frames if total_frames > 0 else None, unit="frame", desc="Tracking")
start_time = time.time()
frame_idx = 0


last_drawn = {}
wrong_way_active_count = 0
total_wrong_way_events = 0

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
        wrong_way_active_count = 0

        if results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                position_history[track_id].append((cx, cy, frame_idx))

                dy, dx = track_net_direction(position_history[track_id])
                if len(position_history[track_id]) >= MIN_HISTORY_FOR_VERDICT and abs(dy) >= MIN_DISPLACEMENT_PX:
                    direction_votes.append(1 if dy > 0 else -1)

    
            if len(direction_votes) > 0:
                dominant_sign = 1 if sum(direction_votes) >= 0 else -1
            else:
                dominant_sign = -1  

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                history = position_history[track_id]
                dy, dx = track_net_direction(history)

                is_wrong = False
                if len(history) >= MIN_HISTORY_FOR_VERDICT and abs(dy) >= MIN_DISPLACEMENT_PX:
                    this_sign = 1 if dy > 0 else -1
   
                    if this_sign != dominant_sign:
                        margin = abs(dy) / max(proc_height, 1)
                        if margin > WRONG_WAY_VOTE_MARGIN * 0.05: 
                            is_wrong = True

                verdict_history[track_id].append(is_wrong)
                smoothed_wrong = sum(verdict_history[track_id]) > len(verdict_history[track_id]) / 2

                last_drawn[track_id] = (x1, y1, x2, y2, smoothed_wrong)

                if smoothed_wrong:
                    wrong_way_active_count += 1
                    if track_id not in wrong_way_ids_seen:
                        wrong_way_ids_seen.add(track_id)
                        total_wrong_way_events += 1

 
    for tid, (x1, y1, x2, y2, is_wrong) in last_drawn.items():
        color = (0, 0, 255) if is_wrong else (0, 220, 0)  
        fill_vehicle(frame, x1, y1, x2, y2, color)

        label = "WRONG WAY" if is_wrong else f"ID {tid}"
        tag_bg = LABEL_COLOR_WRONG if is_wrong else LABEL_COLOR_NORMAL
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 10, y1), tag_bg, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        if is_wrong:
            cv2.putText(frame, "!! ALERT !!", (x1, y2 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    hud_h = 110
    hud_roi = frame[20:20 + hud_h, 20:430]
    dark = np.full_like(hud_roi, (15, 15, 15))
    frame[20:20 + hud_h, 20:430] = cv2.addWeighted(dark, 0.75, hud_roi, 0.25, 0)
    border_color = (0, 0, 255) if wrong_way_active_count > 0 else (0, 255, 200)
    cv2.rectangle(frame, (20, 20), (430, 20 + hud_h), border_color, 1)

    cv2.putText(frame, "WRONG-WAY DETECTOR", (35, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)
    vehicles_now = len(last_drawn)
    cv2.putText(frame, f"Vehicles tracked: {vehicles_now}", (35, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(frame, f"Wrong-way now: {wrong_way_active_count}", (35, 92),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255) if wrong_way_active_count else (0, 220, 0), 1)
    cv2.putText(frame, f"Total wrong-way events: {total_wrong_way_events}", (35, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

    # Flash a big banner across the top when a wrong-way vehicle is active
    if wrong_way_active_count > 0:
        banner_h = 40
        banner_roi = frame[0:banner_h, 0:proc_width]
        red_overlay = np.full_like(banner_roi, (0, 0, 255))
        frame[0:banner_h, 0:proc_width] = cv2.addWeighted(red_overlay, 0.35, banner_roi, 0.65, 0)
        text = "WRONG-WAY VEHICLE DETECTED"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(frame, text, ((proc_width - tw) // 2, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    out.write(frame)

    elapsed = time.time() - start_time
    live_fps = frame_idx / elapsed if elapsed > 0 else 0
    pbar.set_postfix({"vehicles": vehicles_now, "wrong_way": wrong_way_active_count, "fps": f"{live_fps:.1f}"})
    pbar.update(1)

pbar.close()
cap.release()
out.release()

print(f"\n[done] Saved output to: {output_path}")
print(f"[done] Total distinct wrong-way vehicles detected: {total_wrong_way_events}")