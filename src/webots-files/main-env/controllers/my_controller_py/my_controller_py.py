# my_controller_py controller.

from controller import Robot
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import sys
import time
import cv2
import numpy as np
from pupil_apriltags import Detector
import os

# NOTE: normally these are read from env variable
MODEL_PATH = r"yolo11s.pt"  # Update this path
# Confidence threshold for detections
CONFIDENCE_THRESHOLD = 0.45  # Adjust as needed
# Detection interval to manage computational load
DETECTION_INTERVAL = 10  # Perform detection every 10 steps


def load_model(model_pth=MODEL_PATH):
    # === Initialize YOLOv11 Model on CPU ===
    model = None
    try:
        print("Loading YOLOv11 model on CPU...")
        model = YOLO(model_pth)  # Force CPU usage
        model.to("cpu")
        print("YOLOv11 model loaded successfully on CPU.")
    except Exception as e:
        print(f"Error loading YOLOv11 model: {e}")
        sys.exit(1)
    return model


def init_environment(robot):
    """
    The method initialise the robot
    """
    timestep = int(robot.getBasicTimeStep())

    # Initialize the left and right motors
    left_motor = robot.getDevice("left wheel motor")
    right_motor = robot.getDevice("right wheel motor")

    # Set the motors to velocity control mode by setting position to infinity
    left_motor.setPosition(float("inf"))
    right_motor.setPosition(float("inf"))

    # Initialize the camera
    camera = robot.getDevice("camera")  # Ensure the camera device name is correct
    camera.enable(timestep)

    return timestep, camera, left_motor, right_motor


def detection(img, camera, model):
    x_center_int = None
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
                x1, y1, x2, y2 = box.xyxy[
                    0
                ]  # Assuming box.xyxy is a tensor with shape [1, 4]

                # Calculate the center x position
                x_center = (x1 + x2) / 2

                # Convert to integer
                x_center_int = int(
                    x_center.item()
                )  # .item() extracts the value from the tensor

                # Append to the list
                # x_positions.append(x_center_int)

                # print(f"Image 0 x positions: {x_positions}")
        else:
            print("Image 0: []")

    except Exception as e:
        print(f"Error during image processing or detection: {e}")
        # continue  # Continue the loop even if an error occurs
    return x_center_int

def go_to(img, camera, left_motor, right_motor, forward_speed, destination='start'):
    if destination == 'start':
        IMAGE_PATH = r"C:\Users\Marlo\Documents\code\QMUL-Societies\QMES\Unibots-karl\WeBots\many-tags-4.png"
        randomVal = np.random.randint(0, 100000)
        OUTPUT_PATH = f"annotated_image_{randomVal}.jpg"
        VISUALIZE = False  # Set to False if you don't want to display the image

        detect_apriltags(image=img, output_path=OUTPUT_PATH, visualize=VISUALIZE)

def enhance_image(image):
    """
    Enhances the input image to improve AprilTag detection.
    
    Args:
        image (numpy.ndarray): Original BGR image.
    
    Returns:
        numpy.ndarray: Enhanced grayscale image.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Increase contrast using histogram equalization
    gray_eq = cv2.equalizeHist(gray)
    
    # Apply Gaussian blur to reduce noise
    gray_blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)
    
    return gray_blur

def detect_apriltags(image, output_path=None, visualize=False):
    """
    Detects all AprilTags in the given image with enhanced accuracy.

    Args:
        image (numpy.ndarray): Input BGR image.
        output_path (str, optional): Path to save the annotated image. If None, the image won't be saved.
        visualize (bool, optional): If True, displays the annotated image. Defaults to True.
    """
    # Enhance the image to improve detection accuracy
    print(type(image))
    enhanced_gray = enhance_image(image)

    # Initialize the AprilTag detector with optimized parameters
    detector = Detector(
        families='tag36h11',        # Tag family to detect
        nthreads=4,                 # Number of threads to use
        quad_decimate=0.5,          # Lower decimation for higher resolution
        quad_sigma=0.5,             # Apply Gaussian blur
        refine_edges=True,          # Refine tag edges
        decode_sharpening=0.35,     # Increase sharpening
        debug=0                      # Debug mode (0: off, 1: on)
    )

    # Detect AprilTags in the enhanced grayscale image
    tags = detector.detect(enhanced_gray, estimate_tag_pose=False, camera_params=None, tag_size=None)

    # Print the number of detected tags
    print(f"Detected {len(tags)} AprilTag(s) in the image.")

    # Iterate over each detected tag
    for tag in tags:
        # Extract tag information
        tag_id = tag.tag_id
        center = tag.center
        corners = tag.corners

        print(f"\nTag ID: {tag_id}")
        print(f"Tag Center: ({center[0]:.2f}, {center[1]:.2f})")
        print("Tag Corners:")
        for idx, corner in enumerate(corners):
            print(f"  Corner {idx + 1}: ({corner[0]:.2f}, {corner[1]:.2f})")

        # Draw bounding box around the tag
        corners_int = np.int32(corners)  # Correct data type
        cv2.polylines(image, [corners_int], isClosed=True, color=(0, 255, 0), thickness=2)

        # Draw the tag ID near the center
        cv2.putText(image, f"ID: {tag_id}", (int(center[0]) - 10, int(center[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # If no tags are detected, notify the user
    if len(tags) == 0:
        print("No AprilTags detected in the image.")

    # Save the annotated image if an output path is provided
    if output_path:
        cv2.imwrite(output_path, image)
        print(f"\nAnnotated image saved to '{output_path}'.")

    # Display the annotated image if visualization is enabled
    if visualize:
        cv2.imshow('AprilTag Detection', image)
        print("Press any key in the image window to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

def main():
    # === Configuration ===
    # Path to the YOLOv11 model weights
    CHASE_BALL = False
    GO_TO = True
    
    model = load_model(MODEL_PATH)
    # === Initialize Robot ===
    robot = Robot()
    timestep, camera, left_motor, right_motor = init_environment(robot)
    forward_speed = 5.0  # Adjust this value as needed

    print("Controller initialized. Robot is moving forward and capturing images.")

    # Initialize detection counter
    step_count = 0
    prev_x_positions = []
    # Main loop: keep running until the simulation is stopped
    while robot.step(timestep) != -1:
        step_count += 1

        # Capture the image
        img = camera.getImage()

        
        if CHASE_BALL:
            x_positions = []

            if img:
                # Perform detection at specified intervals to manage computational load
                if step_count % DETECTION_INTERVAL == 0:
                    x_center_int = detection(img, camera, model)
                    if x_center_int is not None:
                        x_positions.append(x_center_int)

            # check if x_positions is empty, only overwrite if not empty or last frame was also empty
            if step_count % DETECTION_INTERVAL == 0:
                if not x_positions and prev_x_positions:
                    a = x_positions.copy()
                    x_positions = prev_x_positions.copy()
                    prev_x_positions = a.copy()

            # move robot according to path planning (using x_positions, every frame)
            if x_positions:
                if x_positions[-1] > camera.getWidth() / 2:
                    # move right
                    print("move to the right")
                    left_motor.setVelocity(forward_speed)
                    right_motor.setVelocity(forward_speed * 0.7)
                else:
                    print("move to the left")
                    left_motor.setVelocity(forward_speed * 0.7)
                    right_motor.setVelocity(forward_speed)

            # Optional: Add a sleep or control the loop frequency if needed
            # time.sleep(0.01)
        elif GO_TO:
            go_to(img, camera, left_motor, right_motor, forward_speed, destination='start')

            # movement logic
        
    # Cleanup (optional in Python)
    robot.cleanup()


if __name__ == "__main__":
    main()