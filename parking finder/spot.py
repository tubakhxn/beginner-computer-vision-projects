# dev/creator: tubakhxn

import sys
import cv2
import pickle

if len(sys.argv) > 1:
    video_path = sys.argv[1]
else:
    video_path = "1.mp4"

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print(f"Error: Could not open video file '{video_path}'. Check if it's in this folder!")
    exit()

parking_spots = []
current_box = [0, 0, 0, 0]

def select_spot(event, x, y, flags, param):
    global current_box
    if event == cv2.EVENT_LBUTTONDOWN:
        current_box = [x, y, 0, 0]
    elif event == cv2.EVENT_LBUTTONUP:
        current_box[2] = x - current_box[0]
        current_box[3] = y - current_box[1]

print(f"--- Loaded {video_path} ---")
print("Instructions:")
print("1. Click & drag a box over a spot.")
print("2. Press 'n' if a car is PARKED there.")
print("3. Press 'p' if it is a FREE/EMPTY space.")
print("4. Press 'q' when completely finished.")

cv2.namedWindow("Select Parking Spots", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Select Parking Spots", select_spot)

while True:
    img_copy = frame.copy()

    for x, y, w, h, spot_type in parking_spots:
        color = (0, 0, 255) if spot_type == "parked" else (0, 255, 0)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img_copy, spot_type.upper(), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    if current_box != [0, 0, 0, 0]:
        cv2.rectangle(img_copy, (current_box[0], current_box[1]),
                      (current_box[0] + current_box[2], current_box[1] + current_box[3]), (255, 0, 0), 2)

    cv2.imshow("Select Parking Spots", img_copy)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('n'):
        if current_box != [0, 0, 0, 0]:
            parking_spots.append(current_box + ["parked"])
            print(f"Registered -> PARKED. Total spots: {len(parking_spots)}")
            current_box = [0, 0, 0, 0]
    elif key == ord('p'):
        if current_box != [0, 0, 0, 0]:
            parking_spots.append(current_box + ["free"])
            print(f"Registered -> FREE SPACE. Total spots: {len(parking_spots)}")
            current_box = [0, 0, 0, 0]
    elif key == ord('q'):
        break

with open('parking_positions.pkl', 'wb') as f:
    pickle.dump(parking_spots, f)

cv2.destroyAllWindows()
print(f"\nSuccess! Config saved to 'parking_positions.pkl'. Now run: python car.py 1.mp4")