import cv2
from ultralytics import YOLO
import os
import time

# Configuration parameters
CONFIDENCE_THRESHOLD = 0.5
COLLECT_INFERENCE_DATA = True  # Set to True if you want to save the predicted images
step_count = 0
IMAGE_WIDTH = 320 # 1024
IMAGE_HEIGHT = 176 # 768

current_dir = os.path.dirname(__file__)
# relative_path = os.path.join(current_dir, '..', 'weights', 'real-world-detector.pt')
# relative_path = os.path.join(current_dir, '..', 'weights', 'new-weights', 'yolos-3.onnx')
relative_path = os.path.join(current_dir, '..', 'weights', '0-simple.mnn')
relative_path_inference_output = os.path.join(current_dir, 'object-detection-inference')
MODEL_PATH = os.path.abspath(relative_path)
absolute_path_inference_output = os.path.abspath(relative_path_inference_output)

# Load the YOLO model (update the model path as needed)
model = YOLO(MODEL_PATH) # , task='detect')

# Open the webcam (device 0)
cap = cv2.VideoCapture(0)
print(f"Previous camera settings: width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)} height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)
# Allow time for the camera to adjust the new settings
time.sleep(2)
print(f"Next camera settings: width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)} height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
if not cap.isOpened():
    print("Error: Cannot open webcam.")
    exit()

print("Press 'q' to quit the inference loop.")

while True:
    for _ in range(10):
        cap.grab()
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to retrieve frame. Exiting...")
        break
    print("Frame shape:", frame.shape)

    # Run inference on the current frame
    results = model(frame,
                    conf=CONFIDENCE_THRESHOLD,
                    save=COLLECT_INFERENCE_DATA,
                    project=absolute_path_inference_output,
                    name=f"inference_{step_count}")
    
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
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()