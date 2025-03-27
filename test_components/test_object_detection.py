import cv2
from ultralytics import YOLO
import os

# Configuration parameters
CONFIDENCE_THRESHOLD = 0.5
COLLECT_INFERENCE_DATA = False  # Set to True if you want to save the predicted images
step_count = 0

current_dir = os.path.dirname(__file__)
relative_path = os.path.join(current_dir, '..', 'weights', 'real-world-detector.pt')
MODEL_PATH = os.path.abspath(relative_path)

# Load the YOLO model (update the model path as needed)
model = YOLO(MODEL_PATH)

# Open the webcam (device 0)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam.")
    exit()

print("Press 'q' to quit the inference loop.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to retrieve frame. Exiting...")
        break

    # Run inference on the current frame
    results = model(frame,
                    conf=CONFIDENCE_THRESHOLD,
                    save=COLLECT_INFERENCE_DATA,
                    project=f"predictions/inference_{step_count}.jpg")
    
    # Process the results (assumes single image inference)
    result = results[0]
    boxes = result.boxes  # This is the Boxes object for bounding boxes

    if boxes and len(boxes) > 0:
        for box in boxes:
            # Extract bounding box coordinates (x1, y1, x2, y2)
            # box.xyxy is expected to be a tensor or array; get the first (and only) set of coordinates.
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # Convert to list for easier handling
            print("Box coordinates:", box.xyxy)
            # Calculate the center x position
            x_center = (x1 + x2) / 2
            x_center_int = int(x_center)
            print("Center x coordinate:", x_center_int)
    else:
        print(f"Image {step_count}: []")

    step_count += 1

    # Optionally, display the current frame
    cv2.imshow("Webcam Inference", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()