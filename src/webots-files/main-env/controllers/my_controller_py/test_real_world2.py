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

current_dir = os.path.dirname(__file__)
# relative_path = os.path.join(current_dir, '..', '..', '..', '..', '..', 'weights', 'real-world-detector.pt')
# relative_path = os.path.join(current_dir, '..', '..', '..', '..', '..', 'weights', 'real-world-detector.pt') # needs 640x640
# relative_path = os.path.join(current_dir, '..', '..', '..', '..', '..', 'weights', 'yolos-3.onnx')
relative_path = os.path.join(current_dir, '..', '..', '..', '..', '..', 'weights', '0-simple.mnn') # 640x640
relative_path_inference_output = os.path.join(current_dir, 'object-detection-inference')
MODEL_PATH = os.path.abspath(relative_path)
absolute_path_inference_output = os.path.abspath(relative_path_inference_output)

################
### SETTINGS ###
################

# MODEL_PATH = r"yolo11s.pt" # Use standard model instead    
CONFIDENCE_THRESHOLD = 0.2
DETECTION_FRAME_INTERVAL = 25 # controls how many frames are skipped between apriltag / ball detection is performed
# CAMERA_NAME = "camera"
DISTANCE_THRESHOLD = 650 # 350.0 # Determinisitc works perfect so: blue zone 350 red zone 500 
HOME_IDS = [5, 6] # [23, 0] [5, 6] [11, 12] [17, 18]
# HOME_IDS = [11, 12]
FORWARD_SPEED = 80
ROTATION_SPEED = 50
TURN_RATIO = 0.75 # Drive curves towards ball
# 0.3 * 135 = 40.5
TURN_SPEED_RATIO = -0.7 # 0.7 rotate while moving forward 0.7 # Speed ratio of ROTATION_SPEED - to keep moving towards last april tag position
MAX_MOTOR_SPEED = 250 # Real max speed: 150 | WeBots speed limit:= 6.28 rad/s
ANGLE_GAIN = 12 # simulation is 3 / real: 3-20 *(left 48, right 102)
LAST_TAG_SIDE = None
HOME_TAGS_CENTER_TOLERANCE = 50 # pixels
STEER_GAIN = 0.2 # visual return_home() function
WAIT_FOR_NEXT_FRAME_SECS = 0.7
RECORD_FRAMES_DELAY = 0 # .5 # seconds

TOP_TAGS = set(range(0, 6))      # IDs 0..5
RIGHT_TAGS = set(range(6, 12))   # IDs 6..11
BOTTOM_TAGS = set(range(12, 18)) # IDs 12..17
LEFT_TAGS = set(range(18, 24))   # IDs 18..23

### CAMERA PARAMETERS ###
REMOVE_CAM_BUFFER = 10 # frames to be deleted in the camera buffer, before taking new img
IMAGE_WIDTH = 480
IMAGE_HEIGHT = 640 # 480
FX = 1600 # focal length along x-axis
FY = 1600 # focal length along y-axis
TAG_SIDE_METERS = 0.1 # example: 10cm wide tags

# Global Detector instance for AprilTags
global_detector = Detector(
    families='tag36h11',        # Tag family to detect
    nthreads=4,                 # Number of threads to use
    quad_decimate=0.5,          # Lower decimation for higher resolution
    quad_sigma=0.5,             # Apply Gaussian blur
    refine_edges=True,          # Refine tag edges
    decode_sharpening=0.35,     # Increase sharpening
    debug=0                     # Debug mode (0: off, 1: on)
)

# ROBOT STATES
COMPETITION_START_TIME = 0 # seconds
GO_HOME_TIMER = 25 # seconds
COMPETITION_RETURN = 155 # start go back, and set NO_BALLS_CHASE = TRUE
NO_BALLS_CHASE = False
STOP_COMPETITION_TIMER = 180
COMPETITION = False
CHASE_BALL = False
RETURN_HOME = True

# INDEPENDENT STATES
COLLECT_DATA = False # save frames to disk, to create training data
COLLECT_INFERENCE_DATA = True # save inference data to disk, to check out inference results

# Weird coordinates, because ESU is NOT SUPPORTED in Webots!!!
# Arena corners / Start Positions: 
#   North-West: (-1000, 1000)
#   North-East: (1000, 1000)
#   South-East: (1000, -1000)
#   South-West: (-1000, -1000)

# Distances along each edge (in mm): 150, 300, 300, 500, 300, 300, 150.

TAG_POSITIONS = {
    0:  (-850, 1000),  # Top edge, 150 mm from left (NW) corner
    1:  (-550, 1000),
    2:  (-250, 1000),
    3:  (250, 1000),
    4:  (550, 1000),
    5:  (850, 1000),   # Top edge, 150 mm from right (NE) corner
    6:  (1000, 850),   # Right edge, 150 mm from top (NE) corner
    7:  (1000, 550),
    8:  (1000, 250),
    9:  (1000, -250),
    10: (1000, -550),
    11: (1000, -850),  # Right edge, 150 mm from bottom (SE) corner
    12: (850, -1000),  # Bottom edge, 150 mm from right (SE) corner
    13: (550, -1000),
    14: (250, -1000),
    15: (-250, -1000),
    16: (-550, -1000),
    17: (-850, -1000), # Bottom edge, 150 mm from left (SW) corner
    18: (-1000, -850), # Left edge, 150 mm from bottom (SW) corner
    19: (-1000, -550),
    20: (-1000, -250),
    21: (-1000, 250),
    22: (-1000, 550),
    23: (-1000, 850),  # Left edge, 150 mm from top (NW) corner
}

# SWAPPED (23, 0) to work in april tags directions
HOME_POSITIONS = {
    (23, 0): (-1000, 1000),    # NW corner: midpoint of TAG_POSITIONS[0] and TAG_POSITIONS[23]
    (5, 6): (1000, 1000),      # NE corner: midpoint of TAG_POSITIONS[5] and TAG_POSITIONS[6]
    (11, 12): (1000, -1000),   # SE corner: midpoint of TAG_POSITIONS[11] and TAG_POSITIONS[12]
    (17, 18): (-1000, -1000)   # SW corner: midpoint of TAG_POSITIONS[17] and TAG_POSITIONS[18]
}


def load_model(model_pth=MODEL_PATH):
    ''' Initialize YOLOv11 Model on CPU '''
    model = None
    try:
        print("Loading YOLOv11 model on CPU...")
        model = YOLO(model_pth, task='detect')
        # model.to("cpu")
        print("YOLOv11 model loaded successfully on CPU.")
    except Exception as e:
        print(f"Error loading YOLOv11 model: {e}")
        sys.exit(1)
    # model.eval()
    return model


def bytes_to_numpy(img_data):
    # Check if input is already a numpy array
    if isinstance(img_data, np.ndarray):
        return img_data
    global IMAGE_WIDTH, IMAGE_HEIGHT
    try:
        # Convert the raw image bytes to a NumPy array
        img_array = np.frombuffer(img_data, dtype=np.uint8).reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 3))
        # Make a copy to ensure writeability
        img_rgb = img_array[:, :, :3].copy()
        return img_rgb
    except Exception as e:
        print(f"Error converting bytes to NumPy array: {e}")
        return None



def ball_detection(img, model, step_count):
    global COLLECT_INFERENCE_DATA, CONFIDENCE_THRESHOLD, IMAGE_WIDTH, IMAGE_HEIGHT
    x_center_int = None
    y_center_int = None
    try:
        # Convert the raw image data to a NumPy array
        # img_array = np.frombuffer(img, dtype=np.uint8).reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 3))

        # Convert RGBA to RGB by removing the alpha channel
        img_rgb = img.copy() # img_array[:, :, :3]

        # Create a PIL Image from the NumPy array
        image = Image.fromarray(img_rgb)
        image_np = np.array(image)
        print("Model inference starting...")

        # Run the YOLOv11 model on the image
        # with torch.no_grad():
        results = model(image_np, conf=CONFIDENCE_THRESHOLD, save=COLLECT_INFERENCE_DATA, name=f"inference_{step_count}")
        
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
                y_center = (y1 + y2) / 2

                # Convert to integer
                x_center_int = int(x_center.item())
                y_center_int = int(y_center.item())
        else:
            print("Image 0: []")

    except Exception as e:
        print(f"Error during image processing or detection: {e}")
    return x_center_int, y_center_int


def arrival_routine(wheel_motors):
    """
    Handles the arrival routine when the robot is close enough to home.
    Backs up for 1 second, rotates away from home for 1 second,
    stops the motors, and switches the mode from RETURN_HOME to CHASE_BALL.
    """
    global CHASE_BALL, RETURN_HOME, FORWARD_SPEED, DISTANCE_THRESHOLD

    print(f"Within threshold of {DISTANCE_THRESHOLD} mm. Initiating arrival routine: backing up and rotating away.")

    # Back up for 1 second.
    wheel_motors.setVelocity(-FORWARD_SPEED, -FORWARD_SPEED)
    time.sleep(1.0)  # Back up for 1 second

    # Rotate away from home for 1 second.
    print("Rotating away from home position...")
    wheel_motors.setVelocity(FORWARD_SPEED, -FORWARD_SPEED)
    time.sleep(1.0)  # Rotate for 1 second

    # Stop motors and switch modes.
    wheel_motors.setVelocity(0, 0)
    CHASE_BALL = True
    RETURN_HOME = False
    print("Destination reached. Switching to CHASE_BALL mode.")


def get_destination_coordinate():
    """
    Returns the pre-encoded (x, y) home coordinate based on the global HOME_IDS.
    """
    global HOME_POSITIONS, HOME_IDS
    key = tuple(sorted(HOME_IDS))
    try:
        return HOME_POSITIONS[key]
    except KeyError:
        raise ValueError(f"HOME_IDS {HOME_IDS} do not map to a known corner.")
    

def return_home_deterministic(img_bytes, camera, wheel_motors, step):
    global LAST_TAG_SIDE, ROTATION_SPEED, TURN_SPEED_RATIO, DISTANCE_THRESHOLD, IMAGE_WIDTH, RETURN_HOME, CHASE_BALL, HOME_IDS

    # 1) Convert bytes → NumPy array.
    img_array = bytes_to_numpy(img_bytes)
    if img_array is None:
        print("ERROR: Failed to convert image bytes to a NumPy array.")
        return

    # 2) Detect AprilTags in the image.
    OUTPUT_PATH = f"annotated_image_{step}.jpg"
    detected_tags = detect_apriltags(image=img_array, output_path=OUTPUT_PATH)
    if not detected_tags:
        # No tags detected; fallback to search/spin in place.
        if LAST_TAG_SIDE == "left":
            print("No tags detected; rotating left to search.")
            wheel_motors.setVelocity(TURN_SPEED_RATIO * ROTATION_SPEED, ROTATION_SPEED)
        else:
            print("No tags detected; rotating right to search.")
            wheel_motors.setVelocity(ROTATION_SPEED, TURN_SPEED_RATIO * ROTATION_SPEED)
        return

    # 3) Check if any home tag is "centered" within 30% of the image width.
    image_center_x = IMAGE_WIDTH // 2
    # Define the half-band width as 15% of the image width.
    center_band = 0.25 * IMAGE_WIDTH
    home_tags_in_center = [tag for tag in detected_tags 
                           if tag['id'] in HOME_IDS and abs(tag['center'][0] - image_center_x) < center_band]
    if home_tags_in_center:
        print("Home tag is centered (within 30% of center); moving forward.")
        wheel_motors.setVelocity(FORWARD_SPEED, FORWARD_SPEED)
        return

    # 4) If no home tag is centered, decide a turn based on all detected tags.
    detected_tags_sorted = sorted(detected_tags, key=lambda tag: tag['center'][0])
    mid_index = len(detected_tags_sorted) // 2
    middle_tag = detected_tags_sorted[mid_index]
    middle_id = middle_tag['id']
    
    # Decide turn direction by comparing the middle tag's id with the two HOME_IDS.
    if abs(middle_id - HOME_IDS[0]) < abs(middle_id - HOME_IDS[1]):
        print(f"Middle tag id {middle_id} is closer to {HOME_IDS[0]}")
        print("Turn Right")
        LAST_TAG_SIDE = "right"
        wheel_motors.setVelocity(ROTATION_SPEED, ROTATION_SPEED * TURN_SPEED_RATIO)
    else:
        print(f"Middle tag id {middle_id} is closer to {HOME_IDS[1]} (turn left).")
        print("Turn Left")
        LAST_TAG_SIDE = "left"
        wheel_motors.setVelocity(ROTATION_SPEED * TURN_SPEED_RATIO, ROTATION_SPEED)



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
    global FX, FY, TAG_SIDE_METERS, COLLECT_INFERENCE_DATA, global_detector
    try:
        # Enhance the image to improve detection accuracy
        print(f"Image type before enhancement: {type(image)}")
        print(f"Image writeable before enhancement: {image.flags.writeable}")
        enhanced_gray = enhance_image(image)
        
        # Compute camera parameters (ensure these are of type float)
        cx = image.shape[1] / 2.0  # principal point x
        cy = image.shape[0] / 2.0  # principal point y
        
        # Use the global detector to detect tags
        tags = global_detector.detect(enhanced_gray, estimate_tag_pose=True, camera_params=(FX, FY, cx, cy), tag_size=TAG_SIDE_METERS)

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

            # Extract pose information
            tvec = tag.pose_t  # Translation vector [tx, ty, tz]
            Rmat = tag.pose_R  # Rotation matrix 3x3

            # Calculate distance (in meters)
            distance = np.linalg.norm(tvec)

            # Calculate bearing (angle in the X-Z plane)
            tx, ty, tz = tvec
            bearing = math.atan2(tx, tz)

            # Calculate yaw angle
            yaw = math.atan2(Rmat[1, 0], Rmat[0, 0])

            pose_dict = {
                'distance': distance * 1000.0,  # convert to mm
                'bearing': bearing,
                'yaw': yaw
            }

            detected_tags.append({
                'id': tag_id,
                'center': center,
                'pose': pose_dict
            })

            # Annotate the image
            if not image.flags.writeable:
                image = image.copy()
            corners_int = np.int32(corners)
            cv2.polylines(image, [corners_int], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(image, f"ID: {tag_id}",
                        (int(center[0]) - 10, int(center[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if len(tags) == 0:
            print("No AprilTags detected in the image.")

        if output_path and COLLECT_INFERENCE_DATA:
            cv2.imwrite(output_path, image)
            print(f"\nAnnotated image saved to '{output_path}'.")

        # Sort by the x-coordinate of the center
        sorted_tags = sorted(detected_tags, key=lambda t: t['center'][0])
        print("Detected Tags Ordered by X-Center Position:")
        for t in sorted_tags:
            print(f"ID: {t['id']}, Center=({t['center'][0]:.2f},{t['center'][1]:.2f}), "
                  f"Dist={t['pose']['distance']:.1f}mm, Bearing={t['pose']['bearing']:.2f} rad")

        return sorted_tags
    except Exception as e:
        print(f"Error in detect_apriltags: {e}")
        return None



def estimate_robot_pose(detections):
    """
    Given a list of tag detections (each with a known global TAG_POSITIONS[id]
    and a relative robot->tag transform), estimate the robot's global pose.
    
    For simplicity, we might:
       1) For each detected tag, know the tag's (x_t, y_t).
       2) Use the detection's distance + bearing to guess the robot's position
          (x_r, y_r) = (x_t, y_t) - relative_offset(...)
       3) Possibly average the positions from each detection.
       4) Estimate heading from e.g. the tag's yaw or from multiple detections.
    """
    if not detections:
        return None  # No pose possible

    # Very naive example: average the implied (x_r, y_r) from each tag
    robot_positions = []

    for det in detections:
        tid = det['id']
        if tid not in TAG_POSITIONS:
            continue
        tag_x, tag_y = TAG_POSITIONS[tid]
        # Suppose 'pose' has 'distance' (r) and 'bearing' (b) in *robot* frame
        # so the robot is (r,b) away from the tag in polar coords (tag as origin).
        r = det['pose']['distance']
        b = det['pose']['bearing']

        x_r = tag_x - r * math.cos(b)
        y_r = tag_y - r * math.sin(b)

        robot_positions.append((x_r, y_r))

    # Average them
    if robot_positions:
        x_est = sum(p[0] for p in robot_positions) / len(robot_positions)
        y_est = sum(p[1] for p in robot_positions) / len(robot_positions)
        # Average the yaw values properly over all detections.
        sum_sin = sum(math.sin(det['pose'].get('yaw', 0.0)) for det in detections)
        sum_cos = sum(math.cos(det['pose'].get('yaw', 0.0)) for det in detections)
        theta_est = math.atan2(sum_sin, sum_cos)
        return (x_est, y_est, theta_est)
    else:
        return None


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


def main():
    try:
        # Initialise global variables
        global CHASE_BALL, RETURN_HOME, FORWARD_SPEED, ROTATION_SPEED, TURN_RATIO, TURN_SPEED_RATIO, COMPETITION, COLLECT_DATA, GO_HOME_TIMER, IMAGE_WIDTH, IMAGE_HEIGHT, REMOVE_CAM_BUFFER, WAIT_FOR_NEXT_FRAME_SECS, RECORD_FRAMES_DELAY
        # Initialise local variables
        step_count = 0
        prev_x_positions = []
        print("Starting robot...")
        
        # Load object detection model
        model = load_model(MODEL_PATH)
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
        
        input("Press any key to start the robot...")
        
        # Fine-Tune
        wheel_motors.setVelocity(0, 0)
        wheel_motors.setVelocity(120, 90)
        time.sleep(0.5)
        wheel_motors.setVelocity(120, 90)
        time.sleep(0.5)
        wheel_motors.setVelocity(120, 90)
        time.sleep(4.5)
        wheel_motors.setVelocity(80, -80)
        time.sleep(1.25)
        wheel_motors.setVelocity(80, 80)
        time.sleep(1.2)
        wheel_motors.setVelocity(0, 0)
        
        start_time = time.time()
        chase_start_time = time.time()

        print("4) Starting the main loop.")
        while True:
            current_time = time.time()
            print(f"Timer: {current_time - start_time}")
            if current_time - start_time >= 25:
                print("WE STOPPED!!!!!")
                wheel_motors.setVelocity(0, 0)
                wheel_motors.setVelocity(0, 0)
                exit()
                sys.exit()
            
            time.sleep(RECORD_FRAMES_DELAY)
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
                y_positions = []
                x_center_int, y_center_int = ball_detection(img, model, step_count)
                if x_center_int is not None:
                    x_positions.append(x_center_int)
                
                if not x_positions and prev_x_positions:
                    a = x_positions.copy()
                    x_positions = prev_x_positions.copy()
                    prev_x_positions = a.copy()
                elif not x_positions and not prev_x_positions:
                    print("Forward rotation movement to find ball")
                    wheel_motors.setVelocity(ROTATION_SPEED, ROTATION_SPEED * TURN_SPEED_RATIO)
                
                if x_positions and y_positions:
                    x_y_pair = min(zip(x_positions, y_positions), key=lambda p: p[1])
                    closest_x = x_y_pair[0]
                    if closest_x > IMAGE_WIDTH // 2:
                        print("Move to the right")
                        wheel_motors.setVelocity(FORWARD_SPEED, FORWARD_SPEED * TURN_SPEED_RATIO)
                    else:
                        print("Move to the left")
                        wheel_motors.setVelocity(FORWARD_SPEED * TURN_SPEED_RATIO, FORWARD_SPEED)
                time.sleep(WAIT_FOR_NEXT_FRAME_SECS)
            elif RETURN_HOME:
                return_home_deterministic(img, cap, wheel_motors, step=step_count)
                # return_home(img, wheel_motors, step=step_count, destination_ids=HOME_IDS)
                # return_home_visual_servo(img, camera, left_motor, right_motor, robot,
                #             step=step_count // DETECTION_FRAME_INTERVAL)
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