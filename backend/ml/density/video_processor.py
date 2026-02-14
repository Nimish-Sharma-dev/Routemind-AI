import cv2
import os
import pandas as pd
from ultralytics import YOLO

VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
CONF_THRESHOLD = 0.4

model = YOLO("yolov8n.pt")


def count_vehicles_frame(frame):
    results = model(frame, verbose=False)
    detections = results[0].boxes

    count = 0
    for box in detections:
        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])

        if cls_id in VEHICLE_CLASSES and confidence > CONF_THRESHOLD:
            count += 1

    return count


def process_video(video_path, output_csv="data/yolo_generated.csv", frame_interval=5):
    # Ensure data folder exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ ERROR: Video could not be opened.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = 0
    data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % frame_interval == 0:
            count = count_vehicles_frame(frame)
            time_seconds = frame_number / fps

            data.append({
                "time_seconds": round(time_seconds, 2),
                "vehicle_count": count
            })

            print(f"Time {round(time_seconds,2)}s: {count} vehicles")

        frame_number += 1

    cap.release()

    if not data:
        print("⚠ No data collected.")
        return

    df = pd.DataFrame(data)

    # Append safely (write header only if file doesn't exist)
    write_header = not os.path.exists(output_csv)

    df.to_csv(
        output_csv,
        mode='a',
        header=write_header,
        index=False
    )

    print(f"\nTotal frames processed: {frame_number}")
    print(f"Data collected: {len(data)} rows")
    print(f"Saved data to {output_csv}")


if __name__ == "__main__":
    process_video("backend/ml/density/traffic_video.mp4")
