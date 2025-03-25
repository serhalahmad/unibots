import cv2
import torch
from ultralytics import YOLO

# Load your custom YOLOv11 model
model = YOLO("/home/unibots/unibots/weights/real-world-detector.pt")
print("Model Loaded!")
# Open webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run inference
    results = model(frame)  # Returns a list of Results objects

    # Process results (we only have one frame, so results[0])
    result = results[0]
    boxes = result.boxes  # Get detected bounding boxes

    if boxes:
        for box in boxes:
            # Extract the bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x_center = (x1 + x2) / 2  # Calculate center x position

            # Convert to integer
            x_center_int = int(x_center.item())

            print(f"Bounding Box: {box.xyxy.tolist()}")
            print(f"Center X Position: {x_center_int}")

            # Draw the bounding box on the frame
            # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    else:
        print("No objects detected")

    # Show the frame with detections
    # cv2.imshow("YOLOv11 Live Inference", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
