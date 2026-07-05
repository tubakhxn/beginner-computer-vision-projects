# dev/creator: tubakhxn

import sys
import cv2
import pickle
from ultralytics import YOLO

if len(sys.argv) > 1:
    video_path = sys.argv[1]
else:
    video_path = "1.mp4"

try:
    with open('parking_positions.pkl', 'rb') as f:
        PARKING_SPOTS = pickle.load(f)
    print(f"--> Loaded {len(PARKING_SPOTS)} customized spots from file!")
except FileNotFoundError:
    print("Error: 'parking_positions.pkl' not found. Run spot.py first!")
    exit()

model = YOLO('yolov8n.pt')
spot_statuses = {i: [] for i in range(len(PARKING_SPOTS))}
BUFFER_FRAMES = 3

cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('parking_lot_detected_output.mp4', fourcc, fps, (width, height))

print("--> Speed Hacks Enabled! Processing video fast...")
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + 2)

    if frame_count % 15 == 0:
        print(f"   [Progress] Processing Frame: {frame_count} / {total_frames}...", flush=True)

    results = model(frame, verbose=False)[0]
    detected_vehicles = []

    for box in results.boxes:
        cls = int(box.cls[0])
        if cls in [2, 5, 7]:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detected_vehicles.append([x1, y1, x2 - x1, y2 - y1])

    occupied_count = 0
    empty_count = 0

    for idx, (sx, sy, sw, sh, default_type) in enumerate(PARKING_SPOTS):
        spot_has_car = False

        for (vx, vy, vw, vh) in detected_vehicles:
            if sx < (vx + vw // 2) < sx + sw and sy < (vy + vh // 2) < sy + sh:
                spot_has_car = True
                break

        spot_statuses[idx].append(spot_has_car)
        if len(spot_statuses[idx]) > BUFFER_FRAMES:
            spot_statuses[idx].pop(0)

        detected_active = sum(spot_statuses[idx]) > (BUFFER_FRAMES / 2)

        if default_type == "parked" or detected_active:
            color = (0, 0, 255)
            occupied_count += 1
        else:
            color = (0, 255, 0)
            empty_count += 1

        overlay = frame.copy()
        cv2.rectangle(overlay, (sx, sy), (sx + sw, sy + sh), color, -1)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
        cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), color, 1)

    cv2.rectangle(frame, (30, 30), (420, 160), (0, 0, 0), -1)
    cv2.putText(frame, f"Total Layout Spots: {len(PARKING_SPOTS)}", (50, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Occupied / Engaged: {occupied_count}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, f"Empty / Available: {empty_count}", (50, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)
    out.write(frame)
    out.write(frame)

cap.release()
out.release()
print("\n--> Finished! Video saved as 'parking_lot_detected_output.mp4'")