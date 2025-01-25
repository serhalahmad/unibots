# my_controller_py controller.

from controller import Robot
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import sys
import time

def main():
    # === Configuration ===
    # Path to the YOLOv11 model weights
    MODEL_PATH = r"yolo11s.pt"  # Update this path

    # Confidence threshold for detections
    CONFIDENCE_THRESHOLD = 0.45  # Adjust as needed

    # Detection interval to manage computational load
    DETECTION_INTERVAL = 10  # Perform detection every 10 steps

    # === Initialize YOLOv11 Model on CPU ===
    try:
        print("Loading YOLOv11 model on CPU...")
        model = YOLO(MODEL_PATH) # Force CPU usage
        model.to('cpu')
        print("YOLOv11 model loaded successfully on CPU.")
    except Exception as e:
        print(f"Error loading YOLOv11 model: {e}")
        sys.exit(1)

    # === Initialize Robot ===
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    # Initialize the left and right motors
    left_motor = robot.getDevice('left wheel motor')
    right_motor = robot.getDevice('right wheel motor')

    # Set the motors to velocity control mode by setting position to infinity
    left_motor.setPosition(float('inf'))
    right_motor.setPosition(float('inf'))

    forward_speed = 5.0  # Adjust this value as needed

    # Initialize the camera
    camera = robot.getDevice('camera')  # Ensure the camera device name is correct
    camera.enable(timestep)

    print("Controller initialized. Robot is moving forward and capturing images.")

    # Initialize detection counter
    step_count = 0
    prev_x_positions = []
    # Main loop: keep running until the simulation is stopped
    while robot.step(timestep) != -1:
        step_count += 1

        # Capture the image
        img = camera.getImage()
        
        x_positions = []
        
        if img:
            # Perform detection at specified intervals to manage computational load
            if step_count % DETECTION_INTERVAL == 0:
                try:
                    # Get image dimensions
                    width = camera.getWidth()
                    height = camera.getHeight()

                    # Convert the raw image data to a NumPy array
                    img_array = np.frombuffer(img, dtype=np.uint8).reshape((height, width, 4))

                    # Convert RGBA to RGB by removing the alpha channel
                    img_rgb = img_array[:, :, :3]

                    # Create a PIL Image from the NumPy array
                    image = Image.fromarray(img_rgb)

                    # Optionally, resize the image to reduce processing time
                    # Uncomment the following line to resize
                    # image = image.resize((320, 240))  # Example size

                    # Convert the PIL Image to a format compatible with YOLOv11 (NumPy array)
                    image_np = np.array(image)
                    print("Model inference starting...")

                    # Run the YOLOv11 model on the image
                    results = model(image_np, conf=CONFIDENCE_THRESHOLD)

                    # Initialize x_positions for the single image
                    # x_positions = []

                    # Process and print detected objects
                    result = results[0]  # Since there's only one image
                    boxes = result.boxes  # Boxes object for bounding box outputs
                    if boxes:
                        for box in boxes:
                            # Extract the bounding box coordinates (x1, y1, x2, y2)
                            x1, y1, x2, y2 = box.xyxy[0]  # Assuming box.xyxy is a tensor with shape [1, 4]

                            # Calculate the center x position
                            x_center = (x1 + x2) / 2

                            # Convert to integer
                            x_center_int = int(x_center.item())  # .item() extracts the value from the tensor

                            # Append to the list
                            x_positions.append(x_center_int)

                        print(f"Image 0 x positions: {x_positions}")
                    else:
                        print("Image 0: []")
                    
                except Exception as e:
                    print(f"Error during image processing or detection: {e}")
                    continue  # Continue the loop even if an error occurs
                    
        # check if x_positions is empty, only overwrite if not empty or last frame was also empty
        if step_count % DETECTION_INTERVAL == 0:
            if not x_positions and prev_x_positions:
                a = x_positions.copy()
                x_positions = prev_x_positions.copy()
                prev_x_positions = a.copy()
        
        # move robot according to path planning (using x_positions, every frame)
        if x_positions:
            if x_positions[-1] > camera.getWidth()/2:
                # move right
                print("move to the right")
                left_motor.setVelocity(forward_speed)
                right_motor.setVelocity(forward_speed*0.7)
            else:
                print("move to the left")
                left_motor.setVelocity(forward_speed*0.7)
                right_motor.setVelocity(forward_speed)
            
            
        # Optional: Add a sleep or control the loop frequency if needed
        # time.sleep(0.01)

    # Cleanup (optional in Python)
    robot.cleanup()

if __name__ == '__main__':
    main()