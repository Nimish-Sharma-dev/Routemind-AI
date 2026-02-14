import os
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(BASE_DIR, "test.jpg")

model = YOLO("yolov8n.pt")

VEHICLE_CLASSES = [2, 3, 5, 7]

def count_vehicles(image_path):
    results = model(image_path)
    detections = results[0].boxes

    vehicle_count = 0

    for box in detections:
        cls_id = int(box.cls[0])
        if cls_id in VEHICLE_CLASSES:
            vehicle_count += 1

    return vehicle_count


if __name__ == "__main__":
    count = count_vehicles(IMAGE_PATH)
    print(f"Vehicle Count: {count}")
