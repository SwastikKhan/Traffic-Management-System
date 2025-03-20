!pip install ultralytics opencv-python
import cv2
import numpy as np
from ultralytics import YOLO
from google.colab.patches import cv2_imshow

# -----------------------
# 1. Load YOLOv8 Model
# -----------------------
# Using the lightweight YOLOv8 nano model.
model = YOLO("yolov8n.pt")

# -----------------------
# 2. Define Traffic Thresholds
# -----------------------
high_traffic_threshold = 10  # high traffic if vehicles >= this value
low_traffic_threshold = 3    # low traffic if vehicles <= this value

# -----------------------
# 3. Open Video File
# -----------------------
# Make sure you have uploaded a video file named "traffic.mp4" to Colab.
cap = cv2.VideoCapture("/content/27260-362770008_small.mp4")
if not cap.isOpened():
    print("Error: Could not open video file. Please upload a file named '/content/27260-362770008_small.mp4'.")

# Frame skipping: process every 5th frame
frame_skip = 5
frame_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # end of video

    frame_counter += 1
    # Skip frames if not processing the nth frame
    if frame_counter % frame_skip != 0:
        continue

    # -----------------------
    # Optionally downscale the frame to speed up processing
    # -----------------------
    frame = cv2.resize(frame, (640, 480))

    # -----------------------
    # 4. Run Detection on the Frame
    # -----------------------
    results = model(frame, verbose=False)
    detections = results[0].boxes.data.cpu().numpy()

    # -----------------------
    # 5. Count Vehicles & Annotate Detections
    # -----------------------
    # COCO class IDs for vehicles: 2 (car), 3 (motorcycle), 5 (bus), 7 (truck)
    vehicle_class_ids = [2, 3, 5, 7]
    vehicle_count = 0
    for d in detections:
        class_id = int(d[5])
        if class_id in vehicle_class_ids:
            vehicle_count += 1
            x1, y1, x2, y2 = map(int, d[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{class_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # -----------------------
    # 6. Simulate Traffic Signal Control
    # -----------------------
    if vehicle_count >= high_traffic_threshold:
        signal = "Green (Long Duration)"
    elif vehicle_count <= low_traffic_threshold:
        signal = "Green (Short Duration)"
    else:
        signal = "Green (Standard Duration)"

    cv2.putText(frame, f"Vehicles: {vehicle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2)
    cv2.putText(frame, f"Signal: {signal}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2)

    # -----------------------
    # 7. Display the Frame
    # -----------------------
    cv2_imshow(frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
