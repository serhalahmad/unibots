import numpy as np
from PIL import Image
from ultralytics import YOLO
import sys
import time
import cv2
from pupil_apriltags import Detector
import math
import os
import serial
import torch
import multiprocessing

current_dir = os.path.dirname(__file__)
# relative_path = os.path.join(current_dir, '..', '..', '..', '..', '..', 'weights', 'real-world-detector.pt')
relative_path = os.path.join(current_dir, '..', '..', '..', '..', '..', 'weights', '0-simple.mnn')
MODEL_PATH = os.path.abspath(relative_path)

################
### SETTINGS ###
################

# MODEL_PATH = r"yolo11s.pt" # Use standard model instead    
CONFIDENCE_THRESHOLD = 0.5
DETECTION_FRAME_INTERVAL = 25 # controls how many frames are skipped between apriltag / ball detection is performed
# CAMERA_NAME = "camera"
DISTANCE_THRESHOLD = 500 # 300.0
HOME_IDS = [5, 6]
ROTATION_SPEED = 50
FORWARD_SPEED = 80
MAX_MOTOR_SPEED = 80 # Real max speed: 150 | WeBots speed limit:= 6.28 rad/s
ANGLE_GAIN = 12 # 12 # simulation is 3 / real: 3-20 *(left 48, right 102)
TURN_RATIO = 0.75 # 0.7
HOME_TURN_RATIO = -1 # 0.65
LAST_HOME_ROTATION_DIR = 1 # -1: left, 1: right
LAST_BALL_ROTATION_DIR = 1
COMPETITION_START_TIME = 3 # seconds
GO_HOME_TIMER = 120 # seconds

### CAMERA PARAMETERS ###
REMOVE_CAM_BUFFER = 10 # frames to be deleted in the camera buffer, before taking new img
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
FX = 1600 # focal length along x-axis
FY = 1600 # focal length along y-axis
TAG_SIDE_METERS = 0.1 # example: 10cm wide tags

# ROBOT STATES
COMPETITION = False
CHASE_BALL = False
RETURN_HOME = True

# INDEPENDENT STATES
COLLECT_DATA = True # save frames to disk, to create training data
COLLECT_INFERENCE_DATA = True # save inference data to disk, to check out inference results

TAG_POSITIONS = {
    0:  (150, 2000),
    1:  (450, 2000),
    2:  (750, 2000),
    3:  (1250, 2000),
    4:  (1550, 2000),
    5:  (1850, 2000),
    6:  (2000, 1850),
    7:  (2000, 1550),
    8:  (2000, 1250),
    9:  (2000, 750),
    10: (2000, 450),
    11: (2000, 150),
    12: (1850, 0),
    13: (1550, 0),
    14: (1250, 0),
    15: (750, 0),
    16: (450, 0),
    17: (150, 0),
    18: (0, 150),
    19: (0, 450),
    20: (0, 750),
    21: (0, 1250),
    22: (0, 1550),
    23: (0, 1850),
}


def load_model(model_pth=MODEL_PATH):
    ''' Initialize YOLOv11 Model on CPU '''
    model = None
    try:
        print("Loading YOLOv11 model on CPU...")
        model = YOLO(model_pth)
        model.to("cpu")
        print("YOLOv11 model loaded successfully on CPU.")
    except Exception as e:
        print(f"Error loading YOLOv11 model: {e}")
        sys.exit(1)
    model.eval()
    return model


def bytes_to_numpy(img_bytes):
    """
    Converts image bytes from the camera to a writeable NumPy array.

    Args:
        img_bytes (bytes): Image data from the camera.
        camera (RobotCamera): The camera device.

    Returns:
        numpy.ndarray or None: RGB image as a NumPy array or None if conversion fails.
    """
    global IMAGE_WIDTH, IMAGE_HEIGHT
    try:
        # If they passed us an ndarray directly, just use it
        if isinstance(img_bytes, np.ndarray):
            return img_bytes.copy()

        # Otherwise treat it as raw bytes
        img_array = np.frombuffer(img_bytes, dtype=np.uint8).reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 3))
        # Convert RGBA to RGB by removing the alpha channel and make a copy to ensure writeability
        img_rgb = img_array[:, :, :3].copy()
        return img_rgb
    except Exception as e:
        print(f"Error converting bytes to NumPy array: {e}")
        return None


def ball_detection(img, model, step_count):
    global COLLECT_INFERENCE_DATA, CONFIDENCE_THRESHOLD, IMAGE_WIDTH, IMAGE_HEIGHT
    x_center_int = None
    try:
        # Convert the raw image data to a NumPy array
        img_array = np.frombuffer(img, dtype=np.uint8).reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 3))

        # Convert RGBA to RGB by removing the alpha channel
        img_rgb = img_array[:, :, :3]

        # Create a PIL Image from the NumPy array
        image = Image.fromarray(img_rgb)
        image_np = np.array(image)
        print("Model inference starting...")

        # Run the YOLOv11 model on the image
        with torch.no_grad():
            results = model(image_np, conf=CONFIDENCE_THRESHOLD, save=COLLECT_INFERENCE_DATA, 
                            project=f"inference_{step_count}.jpg")
        
        # Process and print detected objects
        result = results[0] # Since there's only one image
        boxes = result.boxes # Boxes object for bounding box outputs
        if boxes:
            for box in boxes:
                # Extract the bounding box coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0]
                print(box.xyxy)
                # Calculate the center x position
                x_center = (x1 + x2) / 2

                # Convert to integer
                x_center_int = int(x_center.item())
        else:
            print("Image 0: []")

    except Exception as e:
        print(f"Error during image processing or detection: {e}")
    return x_center_int


def get_destination_coordinate(destination_ids):
    """
    Given two tag IDs that define a corner, return the (x,y) for 'home'.
    One simple approach is just to average the corner tags' coordinates.
    """
    if len(destination_ids) != 2:
        raise ValueError("destination_ids must have exactly two elements")
    tid1, tid2 = destination_ids
    (x1, y1) = TAG_POSITIONS[tid1]
    (x2, y2) = TAG_POSITIONS[tid2]
    # For a “corner,” often these two tags are close, so just take midpoint:    
    return ((x1 + x2)/2.0, (y1 + y2)/2.0)


def return_home(img_bytes, wheel_motors, step, destination_ids=[0, 23]):
    global CHASE_BALL, RETURN_HOME, DISTANCE_THRESHOLD, FORWARD_SPEED, ROTATION_SPEED, MAX_MOTOR_SPEED, ANGLE_GAIN, LAST_HOME_ROTATION_DIR, HOME_TURN_RATIO
    print(f">>> return_home STEP {step}: enter")
    # 1) Convert bytes -> NumPy array
    img_array = bytes_to_numpy(img_bytes)
    print(f">>> return_home STEP {step}: after bytes_to_numpy, img_array is", type(img_array))
    if img_array is None:
        print("Failed to convert image bytes to NumPy array.")
        return

    # 2) Detect AprilTags in the image
    OUTPUT_PATH = f"annotated_image_{step}.jpg"
    
    print(f">>> return_home STEP {step}: about to safe_detect_apriltags")
    detected_tags = safe_detect_apriltags(img_array, OUTPUT_PATH)
    print(f">>> return_home STEP {step}: safe_detect_apriltags returned {len(detected_tags)} tags")
    
    
    # print(f">>> return_home STEP {step}: about to detect_apriltags")
    # try:
    #     detected_tags = detect_apriltags(image=img_array, output_path=OUTPUT_PATH)
    #     print(f">>> return_home STEP {step}: detect_apriltags returned {len(detected_tags)} tags")
    # except Exception as e:
    #     print(f">>> return_home STEP {step}: detect_apriltags threw: {e}")
    #     detected_tags = []
    
    if not detected_tags:
        print("No tags detected; rotating in place to find tags.")
        # Rotate in the last known direction
        if LAST_HOME_ROTATION_DIR == 1:
            print("- MOVE RIGHT (default rotation: no tags detected)")
            wheel_motors.setVelocity( ROTATION_SPEED, ROTATION_SPEED * HOME_TURN_RATIO)  # rotate right
        else:
            print("- MOVE LEFT (default rotation: no tags detected)")
            wheel_motors.setVelocity( ROTATION_SPEED * HOME_TURN_RATIO,  ROTATION_SPEED)  # rotate left
        return

    # 3) Estimate the robot pose (x, y, theta) in mm and radians
    print(f">>> return_home STEP {step}: about to estimate_robot_pose")
    pose = estimate_robot_pose(detected_tags)
    print(f">>> return_home STEP {step}: estimate_robot_pose returned", pose)
    if pose is None:
        print("Could not estimate robot pose from detections.")
        # Rotate in the last known direction
        if LAST_HOME_ROTATION_DIR == 1:
            print("- MOVE RIGHT (default rotation: pose wasn't estimated)")
            wheel_motors.setVelocity( ROTATION_SPEED, ROTATION_SPEED * HOME_TURN_RATIO)
        else:
            print("- MOVE LEFT (default rotation: pose wasn't estimated)")
            wheel_motors.setVelocity( ROTATION_SPEED * HOME_TURN_RATIO, ROTATION_SPEED)
        return
    robot_x, robot_y, robot_theta = pose
    print(f"Robot estimated at x={robot_x:.1f}, y={robot_y:.1f}, θ={math.degrees(robot_theta):.1f}°")

    # 4) Get the home (corner) coordinates from the 2 destination IDs
    home_x, home_y = get_destination_coordinate(destination_ids)
    print(f"Home destination is at x={home_x:.1f}, y={home_y:.1f}")

    # 5) Compute the distance and heading error
    dx = home_x - robot_x
    dy = home_y - robot_y
    distance_to_home = math.hypot(dx, dy)

    print(f"Distance to home = {distance_to_home:.1f} mm")
    if distance_to_home < DISTANCE_THRESHOLD:
        # 1) Move backwards for 1 second
        print(f"Within {DISTANCE_THRESHOLD}mm threshold. Moving backwards...")
        wheel_motors.setVelocity(-FORWARD_SPEED, -FORWARD_SPEED)
        time.sleep(1.0)

        # 2) Rotate away from the home position for 1 second
        print("Rotating away from home corner...")
        wheel_motors.setVelocity(ROTATION_SPEED, -ROTATION_SPEED)
        start = time.time()
        time.sleep(1.0)

        # 3) Switch back to CHASE_BALL mode
        print("Moving forward to chase balls...")
        wheel_motors.setVelocity(FORWARD_SPEED, FORWARD_SPEED)
        CHASE_BALL = True
        RETURN_HOME = False
        print("DESTINATION ARRIVED, switching to CHASE_BALL mode.")
        return

    # Otherwise, compute heading error and drive
    target_angle = math.atan2(dy, dx)
    angle_diff = (target_angle - robot_theta + math.pi) % (2*math.pi) - math.pi

    # 6) Turn while driving forward with P-control
    base_speed = min(FORWARD_SPEED, MAX_MOTOR_SPEED)
    turn = ANGLE_GAIN * angle_diff

    # Flip sign if needed so positive angle => turn left
    left_speed = base_speed - turn
    right_speed = base_speed + turn

    left_speed = max(-MAX_MOTOR_SPEED, min(MAX_MOTOR_SPEED, left_speed))
    right_speed = max(-MAX_MOTOR_SPEED, min(MAX_MOTOR_SPEED, right_speed))

    wheel_motors.setVelocity(left_speed, right_speed + 10)
    
    # Update last rotation direction for next time no tags are seen
    if turn > 0:
        print("MOVE LEFT - tags detected")
        LAST_HOME_ROTATION_DIR = -1   # last adjustment was a left turn
        print("next default rotation - LEFT")
    elif turn < 0:
        print("MOVE RIGHT - tags detected")
        LAST_HOME_ROTATION_DIR = 1    # last adjustment was a right turn
        print("next default rotation - RIGHT")

    print(f"Distance to home = {distance_to_home:.1f} mm, angle diff = {math.degrees(angle_diff):.1f}°")
    print(f"Setting left={left_speed:.2f}, right={right_speed:.2f}")


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


def detect_apriltags(image, output_path=None):
    """
    Detects all AprilTags in the given image with enhanced accuracy.

    Args:
        image (numpy.ndarray): Input RGB image.
        output_path (str, optional): Path to save the annotated image. If None, the image won't be saved.
        visualize (bool, optional): If True, displays the annotated image. Defaults to False.

    Returns:
        list of dict: List containing detected tags with their IDs and center positions, ordered by x-coordinate.
    """
    global FX, FY, TAG_SIDE_METERS, COLLECT_INFERENCE_DATA
    
    # Enhance the image to improve detection accuracy
    print(f"Image type before enhancement: {type(image)}")
    print(f"Image writeable before enhancement: {image.flags.writeable}")
    enhanced_gray = enhance_image(image)

    # Initialize the AprilTag detector with optimized parameters
    try:
        detector = Detector(
            families='tag36h11',        # Tag family to detect
            nthreads=4,                 # Number of threads to use
            quad_decimate=0.5,          # Lower decimation for higher resolution
            quad_sigma=0.5,             # Apply Gaussian blur
            refine_edges=True,          # Refine tag edges
            decode_sharpening=0.35,     # Increase sharpening
            debug=0                      # Debug mode (0: off, 1: on)
        )
        
        cx = image.shape[1] / 2 # principal point x
        cy = image.shape[0] / 2 # principal point y
        print(">>> detect_apriltags: about to call detector.detect()")
        tags = detector.detect(enhanced_gray, estimate_tag_pose=True, camera_params=(FX, FY, cx, cy), tag_size=TAG_SIDE_METERS)
        print(f">>> detect_apriltags: returned {len(tags)} tags")
        
    except Exception as e:
        print("caught error: {e}")
        tags = []

    print(f"Detected {len(tags)} AprilTag(s) in the image.")

    detected_tags = []
    for tag in tags:
        tag_id = tag.tag_id
        center = tag.center
        corners = tag.corners

        print(f"\nTag ID: {tag_id}")
        print(f"Tag Center: ({center[0]:.2f}, {center[1]:.2f})")
        print("Tag Corners:")
        for idx, corner in enumerate(corners):
            print(f"  Corner {idx + 1}: ({corner[0]:.2f}, {corner[1]:.2f})")

        # ----------------------------
        # EXTRACT THE POSE INFORMATION
        # ----------------------------
        # pose_t is a 3x1 vector [tx, ty, tz] in *meters*
        # pose_R is a 3x3 rotation matrix
        tvec = tag.pose_t  # e.g. np.array([tx, ty, tz])
        Rmat = tag.pose_R  # 3x3 numpy array

        # 1) Distance = magnitude of translation
        distance = np.linalg.norm(tvec)

        # 2) Bearing:
        #    Bearing is the left/right angle. Typically:
        #      x-axis is right, y-axis is down, z-axis is forward from camera
        #      So a simple approximation is:
        #        bearing = atan2(tx, tz)
        #      i.e. angle in the X-Z plane.  *Check signs carefully!*
        tx, ty, tz = tvec
        bearing = math.atan2(tx, tz)

        # 3) Yaw from rotation matrix:
        #    This can vary depending on your definitions of pitch/roll/yaw.
        #    One common approach if Z is forward and X is right:
        #      yaw = arctan2(Rmat[1,0], Rmat[0,0]) 
        #    But verify which axis you call “yaw.” 
        yaw = math.atan2(Rmat[1,0], Rmat[0,0])

        # Store a 'pose' dict so that estimate_robot_pose(...) can use it
        pose_dict = {
            'distance': distance,  # in meters or convert to mm
            'bearing': bearing,
            'yaw': yaw
        }

        # Convert to mm if your TAG_POSITIONS dictionary is in mm
        pose_dict['distance'] *= 1000.0

        detected_tags.append({
            'id': tag_id,
            'center': center,
            'pose': pose_dict
        })

        # Performance increasement: if COLLECT_INFERENCE_DATA:
        if not image.flags.writeable:
            image = image.copy()
        corners_int = np.int32(corners)
        cv2.polylines(image, [corners_int], isClosed=True,
                      color=(0, 255, 0), thickness=2)
        cv2.putText(image, f"ID: {tag_id}",
                    (int(center[0]) - 10, int(center[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if len(tags) == 0:
        print("No AprilTags detected in the image.")

    if output_path and COLLECT_INFERENCE_DATA:
        cv2.imwrite(output_path, image)
        print(f"\nAnnotated image saved to '{output_path}'.")

    # Sort by X-center just like you had
    sorted_tags = sorted(detected_tags, key=lambda t: t['center'][0])
    print("Detected Tags Ordered by X-Center Position:")
    for t in sorted_tags:
        print(f"ID: {t['id']}, Center=({t['center'][0]:.2f},{t['center'][1]:.2f}), "
              f"Dist={t['pose']['distance']:.1f}mm, Bearing={t['pose']['bearing']:.2f} rad")

    return sorted_tags

# Run AprilTag detection in a child process so a segfault only kills the child
def safe_detect_apriltags(image, output_path=None, timeout=2.0):
    """Call detect_apriltags in a subprocess; on crash or timeout return empty list."""       
    def worker(img, out, q):
        # ensure we save annotated images exactly as the original function does
        global COLLECT_INFERENCE_DATA
        try:
            tags = detect_apriltags(img, output_path=out)
        except Exception:
            tags = []
        q.put(tags)

    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=worker, args=(image, output_path, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        print("safe_detect_apriltags: child hung or crashed, skipping this frame.")
        return []
    try:
        return q.get_nowait()
    except Exception:
        return []

def estimate_robot_pose(detections):
    if not detections:
        return None

    # average X,Y exactly as you have:
    robot_positions = []
    for det in detections:
        tid = det['id']
        if tid not in TAG_POSITIONS: continue
        tx, ty = TAG_POSITIONS[tid]
        r = det['pose']['distance']
        b = det['pose']['bearing']
        x_r = tx - r * math.cos(b)
        y_r = ty - r * math.sin(b)
        robot_positions.append((x_r, y_r))

    x_est = sum(p[0] for p in robot_positions) / len(robot_positions)
    y_est = sum(p[1] for p in robot_positions) / len(robot_positions)

    # circular‐mean the yaws:
    sin_sum = sum(math.sin(det['pose']['yaw']) for det in detections)
    cos_sum = sum(math.cos(det['pose']['yaw']) for det in detections)
    theta_est = math.atan2(sin_sum, cos_sum)

    return (x_est, y_est, theta_est)

class Motor:
    def __init__(self, name, ser):
        self.name = name
        self.ser = ser
        # Initialize motor hardware here

    def setVelocity(self, left_velocity, right_velocity):
        # Send PWM signal or command to motor driver
        print(f"{self.name} for left wheels to {left_velocity} and right wheels to {right_velocity}")
        # Example: "setVelocity 40 75" (left motor 40, right motor 75)
        command = str(self.name) + " " + str(left_velocity) + " " + str(right_velocity) + '\n'
        print(command)
        self.ser.write(command.encode())


def init_real_environment(ser):
    global IMAGE_WIDTH, IMAGE_HEIGHT
    # Initialize the camera using OpenCV
    cap = cv2.VideoCapture(0)  # Open the default camera (change index if needed)
    if not cap.isOpened():
        raise Exception("Could not open video device")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)

    # Create motor objects
    wheel_motors = Motor('setVelocity', ser)

    return cap, wheel_motors


def cleanup(cap):
    cap.release()
    cv2.destroyAllWindows()
    

def get_image(cap, remove_buffer):
    # Flush the buffer by grabbing a few frames
    for _ in range(remove_buffer):
        cap.grab()
    ret, frame = cap.read()
    if not ret:
        print("Warning: Failed to capture image")
        return None
    return frame


def save_image(frame, step_count):
    # Define the folder path
    folder = "main_loop_frames"
    
    # Create the folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Define the file name and path
    filename = f"{folder}/img_{step_count}.png"
    
    # Save the image
    cv2.imwrite(filename, frame)
    print(f"Image saved as {filename}")

def convert_model_to_torchtrace(model, input_example):
    traced_model = torch.jit.trace(model, input_example)
    return traced_model

def convert_model_to_torchscript(model):
    scripted_model = torch.jit.script(model)
    return scripted_model

def main():
    try:
        # Initialise global variables
        global CHASE_BALL, RETURN_HOME, FORWARD_SPEED, ROTATION_SPEED, TURN_RATIO, COMPETITION, COLLECT_DATA, GO_HOME_TIMER, IMAGE_WIDTH, IMAGE_HEIGHT, REMOVE_CAM_BUFFER, LAST_BALL_ROTATION_DIR
        # Initialise local variables
        step_count = 0
        prev_x_positions = []
        print("Starting robot...")
        
        # Load object detection model
        model = YOLO(MODEL_PATH, task='detect')
        print("1) Object detection model loaded successfully.")
        
        ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
        print("2) Serial initialized successfully!")
        
        if COMPETITION:
            print(f"    Competition mode enabled - waiting for {COMPETITION_START_TIME} seconds.")
        
        cap, wheel_motors = init_real_environment(ser)
        print("3) Robot hardware initialized successfully.")
        
        if COMPETITION:
            time.sleep(COMPETITION_START_TIME)
            print(f"    {COMPETITION_START_TIME} seconds have passed. Starting the competition.")
        
        chase_start_time = time.time()
        delay = 0.5  # seconds
        
        print("4) Starting the main loop.")
        while True:
            time.sleep(delay)
            step_count += 1
            
            img = get_image(cap, REMOVE_CAM_BUFFER)
            
            if COLLECT_DATA:
                if img is not None:
                    print("saved img")
                    save_image(img, step_count)
                else:
                    print("Can't save image, cause frame is not recorded correctly!")
            
            if CHASE_BALL and (time.time() - chase_start_time >= GO_HOME_TIMER):
                CHASE_BALL = False
                RETURN_HOME = True
                print(f"    {GO_HOME_TIMER} seconds elapsed in BALL_CHASE mode. Switching to RETURN_HOME mode.")
            
            if CHASE_BALL:
                x_positions = []
                x_center_int = ball_detection(img, model, step_count)
                if x_center_int is not None:
                    x_positions.append(x_center_int)
                
                if not x_positions and prev_x_positions:
                    a = x_positions.copy()
                    x_positions = prev_x_positions.copy()
                    prev_x_positions = a.copy()
                elif not x_positions and not prev_x_positions:
                    print("No ball detected; rotating in place to find ball.")
                    # Rotate in the last known ball-search direction
                    if LAST_BALL_ROTATION_DIR == 1:
                        wheel_motors.setVelocity( ROTATION_SPEED, -ROTATION_SPEED)  # rotate right
                    else:
                        wheel_motors.setVelocity(-ROTATION_SPEED,  ROTATION_SPEED)  # rotate left
                
                if x_positions:
                    if x_positions[-1] > IMAGE_WIDTH / 2:
                        print("Move to the right")
                        wheel_motors.setVelocity(FORWARD_SPEED, FORWARD_SPEED * TURN_RATIO)
                        LAST_BALL_ROTATION_DIR = 1
                    else:
                        print("Move to the left")
                        wheel_motors.setVelocity(FORWARD_SPEED * TURN_RATIO, FORWARD_SPEED)
                        LAST_BALL_ROTATION_DIR = -1
            elif RETURN_HOME:
                # Skip or wrap any error so we never crash on a bad frame
                if img is None:
                    print("No frame available; skipping RETURN_HOME this iteration.")
                else:
                    try:
                        # convert the OpenCV BGR ndarray into raw bytes for return_home
                        # return_home(img.tobytes(), wheel_motors, step=step_count, destination_ids=HOME_IDS)
                        print(f">>> main loop: about to return_home at step {step_count}")
                        return_home(img.tobytes(), wheel_motors, step=step_count, destination_ids=HOME_IDS)
                        print(f">>> main loop: returned from return_home at step {step_count}")
                    except Exception as e:
                        print(f"Error in return_home: {e}; skipping this frame.")
                    if CHASE_BALL:
                        chase_start_time = time.time()
                        print("Returned home. Switching back to BALL_CHASE mode and resetting timer.")
    
    except Exception as e:
        print(f"Error occurred: {e}")
        wheel_motors.setVelocity(0, 0)
        print("Robot stopped for safety.")
    finally:
        print("Robot stopped due to Crtl + C")
        cleanup(cap)

if __name__ == "__main__":
    main()