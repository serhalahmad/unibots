import cv2
import time
import os
from ultralytics import YOLO
from PIL import Image

current_dir = os.path.dirname(__file__)
# relative_path = os.path.join(current_dir, '..', 'weights', 'real-world-detector.pt') # needs 640x640
# relative_path = os.path.join(current_dir, '..', 'weights', 'new-weights', 'yolos-3.onnx')
relative_path = os.path.join(current_dir, '..', 'weights', '0-simple.mnn') # 640x640
relative_path_inference_output = os.path.join(current_dir, 'object-detection-inference')
MODEL_PATH = os.path.abspath(relative_path)
absolute_path_inference_output = os.path.abspath(relative_path_inference_output)

# Load the YOLO model (update the model path as needed)
model = YOLO(MODEL_PATH) # , task='detect')

# Open the default camera (index 0)
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# Optionally, set the camera's resolution (note: some cameras may not support exactly 640x640)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

try:
    while True:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break
        print("Frame shape:", frame.shape)
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        w, h = img.size
        l = (w - 480) // 2
        t = (h - 480) // 2
        cropped = img.crop((l, t, l+480, t+480))
        
        # # Resize the frame to ensure it's 640x640 (in case the camera didn't support it directly)
        # w, h = frame.size
        # left = (w-h) // 2
        # top = (h-)
        # frame_resized = cv2.resize(frame, (640, 640))
        print(f"Shape: {cropped.size}")
        # Print the shape of the frame
        # print("Frame shape:", frame_resized.shape)
        results = model(cropped)
        
        if cv2.waitKey(1) == ord('q'):
            break
except KeyboardInterrupt:
    print("Stopped by user")
finally:
    cap.release()
    cv2.destroyAllWindows()
