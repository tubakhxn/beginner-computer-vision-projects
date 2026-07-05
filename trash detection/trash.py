# dev: tubakhxn

import sys
import subprocess
import importlib

REQUIRED = {
    "cv2": "opencv-python",
    "numpy": "numpy",
    "ultralytics": "ultralytics",
    "tqdm": "tqdm",
}

for module_name, pip_name in REQUIRED.items():
    try:
        importlib.import_module(module_name)
    except ImportError:
        print(f"[setup] Installing missing dependency: {pip_name} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pip_name])

import os
import time
import itertools

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else "trash.mp4"

PROCESS_WIDTH = 1280
DETECT_IMGSZ = 960
FRAME_SKIP = 3
MODEL_NAME = "yolov8s-worldv2.pt"
CONF_THRESHOLD = 0.12

ENABLE_TILED_INFERENCE = True
TILE_SIZE = 960
TILE_OVERLAP = 0.25
TILE_NMS_IOU = 0.45

TRASH_CLASSES = [
    "plastic bag",
    "plastic bottle",
    "white paper on the ground",
    "torn paper debris",
    "newspaper on the ground",
    "flattened cardboard",
    "aluminum can",
    "food wrapper",
    "cup",
    "scattered garbage",
    "litter on the road",
    "trash debris",
]

TRASH_LABEL = "TRASH"
TRASH_COLOR = (0, 0, 255)

MASK_ALPHA = 0.42
LABEL_TEXT_COLOR = (255, 255, 255)

IOU_MATCH_THRESHOLD = 0.30
MIN_CONFIRM_HITS = 3
MAX_MISSED_CYCLES = 6
SMOOTH_ALPHA = 0.55

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
output_path = f"{base_name}_trashdetect.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (proc_width, proc_height))

print("[setup] Loading YOLO-World model...")
model = YOLO(MODEL_NAME)
model.set_classes(TRASH_CLASSES)

try:
    import torch
    device = 0 if torch.cuda.is_available() else "cpu"
except ImportError:
    device = "cpu"
print(f"[setup] Running on device: {device}")


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


tracks = []
id_counter = itertools.count(1)
total_confirmed_ever = set()


def update_tracks(detections):
    unmatched_dets = list(range(len(detections)))
    matched_track_ids = set()

    for track in tracks:
        best_iou, best_j = 0.0, -1
        for j in unmatched_dets:
            det_box = detections[j][0]
            score = iou(track["box"], det_box)
            if score > best_iou:
                best_iou, best_j = score, j
        if best_iou >= IOU_MATCH_THRESHOLD:
            det_box, det_cat, det_conf = detections[best_j]
            ox1, oy1, ox2, oy2 = track["box"]
            nx1, ny1, nx2, ny2 = det_box
            a = SMOOTH_ALPHA
            track["box"] = (
                a * ox1 + (1 - a) * nx1,
                a * oy1 + (1 - a) * ny1,
                a * ox2 + (1 - a) * nx2,
                a * oy2 + (1 - a) * ny2,
            )
            track["category"] = det_cat
            track["conf"] = det_conf
            track["hits"] += 1
            track["missed"] = 0
            unmatched_dets.remove(best_j)
            matched_track_ids.add(track["id"])

    for track in tracks:
        if track["id"] not in matched_track_ids:
            track["missed"] += 1

    tracks[:] = [t for t in tracks if t["missed"] <= MAX_MISSED_CYCLES]

    for j in unmatched_dets:
        det_box, det_cat, det_conf = detections[j]
        tracks.append({
            "id": next(id_counter),
            "box": det_box,
            "category": det_cat,
            "conf": det_conf,
            "hits": 1,
            "missed": 0,
        })


def generate_tiles(width, height, tile_size, overlap):
    stride = int(tile_size * (1 - overlap))
    stride = max(stride, 1)
    xs = list(range(0, max(width - tile_size, 0) + 1, stride)) or [0]
    ys = list(range(0, max(height - tile_size, 0) + 1, stride)) or [0]
    if xs[-1] + tile_size < width:
        xs.append(width - tile_size)
    if ys[-1] + tile_size < height:
        ys.append(height - tile_size)
    for y0 in ys:
        for x0 in xs:
            x1 = min(x0 + tile_size, width)
            y1 = min(y0 + tile_size, height)
            yield (max(x1 - tile_size, 0), max(y1 - tile_size, 0), x1, y1)


def merge_with_nms(detections, iou_thresh):
    if not detections:
        return []
    by_category = {}
    for det in detections:
        by_category.setdefault(det[1], []).append(det)

    merged = []
    for category, dets in by_category.items():
        boxes = [d[0] for d in dets]
        scores = [d[2] for d in dets]
        rects = [[int(x1), int(y1), int(x2 - x1), int(y2 - y1)] for (x1, y1, x2, y2) in boxes]
        keep_idxs = cv2.dnn.NMSBoxes(rects, scores, score_threshold=CONF_THRESHOLD, nms_threshold=iou_thresh)
        if len(keep_idxs) == 0:
            continue
        for i in np.array(keep_idxs).flatten():
            merged.append(dets[i])
    return merged


def run_tiled_detection(native_frame, native_w, native_h, scale_to_proc):
    all_dets = []
    for (x0, y0, x1, y1) in generate_tiles(native_w, native_h, TILE_SIZE, TILE_OVERLAP):
        tile = native_frame[y0:y1, x0:x1]
        if tile.size == 0:
            continue
        results = model.predict(tile, conf=CONF_THRESHOLD, imgsz=DETECT_IMGSZ, device=device, verbose=False)[0]
        if results.boxes is None or len(results.boxes) == 0:
            continue
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        cls_ids = results.boxes.cls.cpu().numpy().astype(int)
        names = results.names
        for box, conf, cls_id in zip(boxes, confs, cls_ids):
            bx1, by1, bx2, by2 = map(float, box)
            gx1, gy1, gx2, gy2 = bx1 + x0, by1 + y0, bx2 + x0, by2 + y0
            gx1, gy1, gx2, gy2 = gx1 * scale_to_proc, gy1 * scale_to_proc, gx2 * scale_to_proc, gy2 * scale_to_proc
            category = names.get(cls_id, "trash") if isinstance(names, dict) else names[cls_id]
            all_dets.append(((gx1, gy1, gx2, gy2), category, float(conf)))

    return merge_with_nms(all_dets, TILE_NMS_IOU)


def run_whole_frame_detection(proc_frame):
    results = model.predict(proc_frame, conf=CONF_THRESHOLD, imgsz=DETECT_IMGSZ, device=device, verbose=False)[0]
    detections = []
    if results.boxes is not None and len(results.boxes) > 0:
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        cls_ids = results.boxes.cls.cpu().numpy().astype(int)
        names = results.names
        for box, conf, cls_id in zip(boxes, confs, cls_ids):
            x1, y1, x2, y2 = map(float, box)
            category = names.get(cls_id, "trash") if isinstance(names, dict) else names[cls_id]
            detections.append(((x1, y1, x2, y2), category, float(conf)))
    return detections


def fill_item(frame, x1, y1, x2, y2, color, alpha=MASK_ALPHA):
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

pbar = tqdm(total=total_frames if total_frames > 0 else None, unit="frame", desc="Scanning")
start_time = time.time()
frame_idx = 0

scale_to_proc = proc_width / orig_width

while True:
    ret, native_frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    frame = cv2.resize(native_frame, (proc_width, proc_height))
    run_detection = (frame_idx % FRAME_SKIP == 0) or frame_idx == 1

    if run_detection:
        if ENABLE_TILED_INFERENCE:
            detections = run_tiled_detection(native_frame, orig_width, orig_height, scale_to_proc)
        else:
            detections = run_whole_frame_detection(frame)

        update_tracks(detections)

    confirmed_now = [t for t in tracks if t["hits"] >= MIN_CONFIRM_HITS]

    for t in confirmed_now:
        x1, y1, x2, y2 = map(int, t["box"])
        fill_item(frame, x1, y1, x2, y2, TRASH_COLOR)

        label = f"{TRASH_LABEL} {t['conf']*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x1, max(y1 - th - 10, 0)), (x1 + tw + 8, y1), TRASH_COLOR, -1)
        cv2.putText(frame, label, (x1 + 4, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, LABEL_TEXT_COLOR, 1, cv2.LINE_AA)

        total_confirmed_ever.add(t["id"])

    hud_w = 340
    hud_h = 90
    hud_x2 = proc_width - 20
    hud_x1 = hud_x2 - hud_w
    hud_y1 = 20
    hud_y2 = hud_y1 + hud_h

    hud_roi = frame[hud_y1:hud_y2, hud_x1:hud_x2]
    dark = np.full_like(hud_roi, (15, 15, 15))
    frame[hud_y1:hud_y2, hud_x1:hud_x2] = cv2.addWeighted(dark, 0.75, hud_roi, 0.25, 0)
    cv2.rectangle(frame, (hud_x1, hud_y1), (hud_x2, hud_y2), (0, 255, 200), 1)

    cv2.putText(frame, "STREET TRASH DETECTOR", (hud_x1 + 15, hud_y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 2)
    cv2.putText(frame, f"Trash visible: {len(confirmed_now)}", (hud_x1 + 15, hud_y1 + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(frame, f"Total found: {len(total_confirmed_ever)}", (hud_x1 + 15, hud_y1 + 74), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

    out.write(frame)

    elapsed = time.time() - start_time
    live_fps = frame_idx / elapsed if elapsed > 0 else 0
    pbar.set_postfix({"items": len(confirmed_now), "total_found": len(total_confirmed_ever), "fps": f"{live_fps:.1f}"})
    pbar.update(1)

pbar.close()
cap.release()
out.release()

print(f"\n[done] Saved output to: {output_path}")
print(f"[done] Total unique trash items detected: {len(total_confirmed_ever)}")