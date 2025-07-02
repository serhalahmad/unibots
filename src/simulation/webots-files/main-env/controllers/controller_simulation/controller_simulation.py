from controller import Robot
import numpy as np
from PIL import Image
from ultralytics import YOLO
import sys
import time
import cv2
from pupil_apriltags import Detector
import math
import os

current_dir = os.path.dirname(__file__)
relative_path = os.path.join(current_dir, '..', '..', '..', '..', '..', '..', 'weights', 'simulation-detector.pt')
MODEL_PATH = os.path.abspath(relative_path)

################
### SETTINGS ###
################

# MODEL_PATH = r"yolo11s.pt" # Use standard model instead    
CONFIDENCE_THRESHOLD = 0.15
DETECTION_FRAME_INTERVAL = 15 # 30 # controls how many frames are skipped between apriltag / ball detection is performed
DATA_COLLECTION_INTERVAL = 15
FRAMES_TILL_TARGET_SWITCH = 50 # How many frames to wait before switching target ball
CAMERA_NAME = "camera"
OBJECT_DETECTION_CLASSES = ["rugby-balls", "ping-pong-ball"]
DISTANCE_THRESHOLD = 500 # 350.0 # Determinisitc works perfect so: blue zone 350 red zone 500 
HOME_IDS = [23, 0] # [23, 0]!!!! OR [5, 6] OR [11, 12] OR [17, 18]
FORWARD_SPEED = 5.0  # Adjust this value as needed
ROTATION_SPEED = 3.5
TURN_SPEED_RATIO = 0.7 # Speed ratio of ROTATION_SPEED - to keep moving towards last april tag position
MAX_MOTOR_SPEED = 6.28 # WeBots speed limit:= 6.28 rad/s
ANGLE_GAIN = 6
TURN_RATIO = 0.7
COMPETITION_START_TIME = 3 # seconds
GO_HOME_TIMER = 15 # seconds
LAST_TAG_SIDE = None
HOME_TAGS_CENTER_TOLERANCE = 50 # pixels
# Original working verison: 1920x1080 | Old: 680x480 | New: 640x640
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 640

TOP_TAGS = set(range(0, 6))      # IDs 0..5
RIGHT_TAGS = set(range(6, 12))   # IDs 6..11
BOTTOM_TAGS = set(range(12, 18)) # IDs 12..17
LEFT_TAGS = set(range(18, 24))   # IDs 18..23

### CAMERA PARAMETERS ###
FX = 1600 # focal length along x-axis
FY = 1600 # focal length along y-axis
TAG_SIDE_METERS = 0.1 # example: 10cm wide tags

# ROBOT STATES
COMPETITION = False
CHASE_BALL = True
RETURN_HOME = False

# INDEPENDENT STATES
COLLECT_DATA = True # save frames to disk, to create training data
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
        model = YOLO(model_pth)
        model.to("cpu")
        print("YOLOv11 model loaded successfully on CPU.")
    except Exception as e:
        print(f"Error loading YOLOv11 model: {e}")
        sys.exit(1)
    return model


def init_environment(robot):
    ''' Initialize the robot variables '''
    timestep = int(robot.getBasicTimeStep())

    # Initialize the left and right motors
    left_motor = robot.getDevice("left wheel motor")
    right_motor = robot.getDevice("right wheel motor")

    # Change position to inf -> to set velocity control mode
    left_motor.setPosition(float("inf"))
    right_motor.setPosition(float("inf"))

    # Initialize the camera
    camera = robot.getDevice(CAMERA_NAME)
    camera.enable(timestep)
    
    return timestep, camera, left_motor, right_motor


def bytes_to_numpy(img_bytes, camera):
    """
    Converts image bytes from the camera to a writeable NumPy array.

    Args:
        img_bytes (bytes): Image data from the camera.
        camera (RobotCamera): The camera device.

    Returns:
        numpy.ndarray or None: RGB image as a NumPy array or None if conversion fails.
    """
    global IMAGE_HEIGHT, IMAGE_WIDTH
    try:
        # Convert the raw image data to a NumPy array
        img_array = np.frombuffer(img_bytes, dtype=np.uint8).reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 4))
        # Convert RGBA to RGB by removing the alpha channel and make a copy to ensure writeability
        img_rgb = img_array[:, :, :3].copy()
        return img_rgb
    except Exception as e:
        print(f"Error converting bytes to NumPy array: {e}")
        return None

def ball_detection(img, camera, model, step_count):
    # 1) RGBA → RGB NumPy array (unchanged)
    img_array = np.frombuffer(img, dtype=np.uint8).reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 4))
    img_rgb   = img_array[:, :, :3]
    image_np  = np.array(Image.fromarray(img_rgb))

    print("Model inference starting...")

    # 2) Run our combined detect+track in one shot, with BoT-SORT (no optical flow)
    results = model.track(
        source = image_np,
        conf   = CONFIDENCE_THRESHOLD,
        persist= True,
        tracker= 'botsort.yaml'        # <— switch to BoT-SORT (no cv2.calcOpticalFlowPyrLK)
    )

    # 3) Grab the first (and only) frame’s result
    result = results[0]
    if COLLECT_INFERENCE_DATA:
        result.save(f"object_detection_predictions/inference_{step_count}.jpg")

    # 4) Build your detections list exactly as before
        detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        id_attr   = getattr(box, "id",   None)
        cls_attr  = getattr(box, "cls",  None)
        conf_attr = getattr(box, "conf", None)

        track_id    = int(id_attr.item())   if id_attr   is not None else -1
        class_index = int(cls_attr.item())  if cls_attr  is not None else -1
        confidence  = float(conf_attr.item()) if conf_attr is not None else 0.0
        x_center    = int(((x1 + x2) / 2).item())

        detected_class = (
            OBJECT_DETECTION_CLASSES[class_index]
            if 0 <= class_index < len(OBJECT_DETECTION_CLASSES)
            else "Unknown"
        )

        print(f"[ID {track_id}] {detected_class} conf={confidence:.2f} "
              f"bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
        detections.append({
            "id":         track_id,
            "class":      detected_class,
            "confidence": confidence,
            "center_x":   x_center,
            "bbox":       (float(x1), float(y1), float(x2), float(y2))
        })
    return detections

def arrival_routine(left_motor, right_motor, robot):
    """
    Handles the arrival routine when the robot is close enough to home.
    Backs up for 1 second, rotates away from home for 1 second,
    stops the motors, and switches the mode from RETURN_HOME to CHASE_BALL.
    """
    global CHASE_BALL, RETURN_HOME, FORWARD_SPEED, DISTANCE_THRESHOLD

    print(f"Within threshold of {DISTANCE_THRESHOLD} mm. Initiating arrival routine: backing up and rotating away.")

    # Back up for 1 second.
    left_motor.setVelocity(-FORWARD_SPEED)
    right_motor.setVelocity(-FORWARD_SPEED)
    start = time.time()
    while (time.time() - start) < 1.0:
        if robot.step(int(robot.getBasicTimeStep())) == -1:
            break

    # Rotate away from home for 1 second.
    print("Rotating away from home position...")
    left_motor.setVelocity(FORWARD_SPEED)
    right_motor.setVelocity(-FORWARD_SPEED)
    start = time.time()
    while (time.time() - start) < 1.0:
        if robot.step(int(robot.getBasicTimeStep())) == -1:
            break

    # Stop motors and switch modes.
    left_motor.setVelocity(0)
    right_motor.setVelocity(0)
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
    

def return_home_deterministic(img_bytes, camera, left_motor, right_motor, robot, step, home_ids=HOME_IDS):
    global LAST_TAG_SIDE, ROTATION_SPEED, TURN_SPEED_RATIO, DISTANCE_THRESHOLD, HOME_TAGS_CENTER_TOLERANCE, IMAGE_WIDTH, RETURN_HOME, CHASE_BALL

    # 1) Convert bytes → NumPy array.
    img_array = bytes_to_numpy(img_bytes, camera)
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
            left_motor.setVelocity(TURN_SPEED_RATIO * ROTATION_SPEED)
            right_motor.setVelocity(ROTATION_SPEED)
        else:
            print("No tags detected; rotating right to search.")
            left_motor.setVelocity(ROTATION_SPEED)
            right_motor.setVelocity(TURN_SPEED_RATIO * ROTATION_SPEED)
        return

    home_tags = [tag for tag in detected_tags if tag['id'] in home_ids]
    for tag in home_tags:
        if tag['pose']['distance'] < DISTANCE_THRESHOLD:
            print("Arrived Home!")
            print("Arrived Home!")
            print("Arrived Home!")
            left_motor.setVelocity(0)
            right_motor.setVelocity(0)
            RETURN_HOME = False
            CHASE_BALL = True
            return

    # 3) Check if any home tag is centered.
    image_center_x = IMAGE_WIDTH // 2
    home_tags_in_center = [tag for tag in detected_tags 
                           if tag['id'] in home_ids and abs(tag['center'][0] - image_center_x) < HOME_TAGS_CENTER_TOLERANCE]
    if home_tags_in_center:
        print("Home tag is centered; moving forward.")
        left_motor.setVelocity(FORWARD_SPEED)
        right_motor.setVelocity(FORWARD_SPEED)
        return

    # 4) No centered home tag; use all detected tags to decide turn direction.
    # Sort all tags by the x-coordinate of their center.
    detected_tags_sorted = sorted(detected_tags, key=lambda tag: tag['center'][0])
    mid_index = len(detected_tags_sorted) // 2
    middle_tag = detected_tags_sorted[mid_index]
    middle_id = middle_tag['id']
    
    # Decide turn direction by comparing the middle tag's id with the two home_ids.
    if abs(middle_id - home_ids[0]) < abs(middle_id - home_ids[1]):
        print(f"Middle tag id {middle_id} is closer to {home_ids[0]} (turn right).")
        LAST_TAG_SIDE = "right"
        left_motor.setVelocity(ROTATION_SPEED)
        right_motor.setVelocity(ROTATION_SPEED * TURN_SPEED_RATIO)
    else:
        print(f"Middle tag id {middle_id} is closer to {home_ids[1]} (turn left).")
        LAST_TAG_SIDE = "left"
        left_motor.setVelocity(ROTATION_SPEED * TURN_SPEED_RATIO)
        right_motor.setVelocity(ROTATION_SPEED)



def return_home_visual_servo(img_bytes, camera, left_motor, right_motor, robot, step, home_ids=HOME_IDS):
    """
    Visual-servo approach to drive home without relying solely on detecting the home corner tags.
    1) If home tag(s) are visible, servo on them directly.
    2) Otherwise, servo on whichever side tags are visible (top/right/bottom/left).
       Because the arena is structured, even partial side info can guide the robot
       toward the home corner.
    """
    global LAST_TAG_SIDE, ROTATION_SPEED, TURN_SPEED_RATIO, DISTANCE_THRESHOLD

    # 1) Convert bytes → NumPy array.
    img_array = bytes_to_numpy(img_bytes, camera)
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
            left_motor.setVelocity(TURN_SPEED_RATIO * ROTATION_SPEED)
            right_motor.setVelocity(ROTATION_SPEED)
        else:
            print("No tags detected; rotating right to search.")
            left_motor.setVelocity(ROTATION_SPEED)
            right_motor.setVelocity(TURN_SPEED_RATIO * ROTATION_SPEED)
        return

    # 3) Check for home tags first.
    home_tags = [tag for tag in detected_tags if tag['id'] in home_ids]

    # If we found the home tags, we can servo on them directly.
    if home_tags:
        print("Home tag(s) detected. Steering directly toward home corner.")
        tags_to_servo = home_tags
    else:
        # 4) Otherwise, we do a fallback approach using side tags.
        # Determine which side (or sides) we see the most.
        side_dict = {"top": [], "right": [], "bottom": [], "left": []}
        for tag in detected_tags:
            tid = tag['id']
            if tid in TOP_TAGS:
                side_dict["top"].append(tag)
            elif tid in RIGHT_TAGS:
                side_dict["right"].append(tag)
            elif tid in BOTTOM_TAGS:
                side_dict["bottom"].append(tag)
            elif tid in LEFT_TAGS:
                side_dict["left"].append(tag)

        # Pick whichever side has the most tags detected (simple heuristic).
        best_side = max(side_dict.keys(), key=lambda s: len(side_dict[s]))
        tags_to_servo = side_dict[best_side]

        if not tags_to_servo:
            # If for some reason all sides are empty (unlikely if we have tags),
            # fallback to spinning in place.
            print("No home corner tags or side tags recognized. Searching...")
            if LAST_TAG_SIDE == "left":
                left_motor.setVelocity(TURN_SPEED_RATIO * ROTATION_SPEED)
                right_motor.setVelocity(ROTATION_SPEED)
            else:
                left_motor.setVelocity(ROTATION_SPEED)
                right_motor.setVelocity(TURN_SPEED_RATIO * ROTATION_SPEED)
            return
        else:
            print(f"No home tags visible. Using {best_side.upper()} side tags to navigate.")

    # 5) (Optional) Check average distance from the tags_to_servo if 'dist' is available.
    # If it's below threshold, call arrival_routine().
    # E.g., if all tags in tags_to_servo have "dist" info:
    if all("dist" in tag for tag in tags_to_servo):
        avg_distance = sum(tag["dist"] for tag in tags_to_servo) / len(tags_to_servo)
        print(f"Average distance to corner/side: {avg_distance:.1f} mm")
        if avg_distance < DISTANCE_THRESHOLD:
            arrival_routine(left_motor, right_motor, robot)
            return

    # 6) Visual servo on whichever tags we ended up with (home or side).
    avg_x = sum(tag['center'][0] for tag in tags_to_servo) / len(tags_to_servo)
    image_center_x = img_array.shape[1] / 2.0
    error_x = avg_x - image_center_x

    # Update LAST_TAG_SIDE based on the horizontal error.
    if error_x < 0:
        LAST_TAG_SIDE = "left"
    else:
        LAST_TAG_SIDE = "right"

    # Simple proportional controller on the horizontal error.
    STEER_GAIN = 0.2  # Tune as needed.
    turn_correction = STEER_GAIN * error_x

    # Drive forward at a constant speed.
    FORWARD_SPEED = 6.0
    left_speed = FORWARD_SPEED - turn_correction
    right_speed = FORWARD_SPEED + turn_correction

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)

    print(
        f"Visual Servo: error_x={error_x:.2f}, turn={turn_correction:.2f}, "
        f"left={left_speed:.2f}, right={right_speed:.2f}"
    )


def return_home(img_bytes, camera, left_motor, right_motor, robot, step, destination_ids=[0, 23]):
    global CHASE_BALL, RETURN_HOME, DISTANCE_THRESHOLD, FORWARD_SPEED
    global ROTATION_SPEED, MAX_MOTOR_SPEED, ANGLE_GAIN, LAST_TAG_SIDE, TURN_SPEED_RATIO

    # 1) Convert bytes → NumPy array.
    img_array = bytes_to_numpy(img_bytes, camera)
    if img_array is None:
        print("ERROR: Failed to convert image bytes to a NumPy array.")
        return

    # 2) Detect AprilTags in the image.
    OUTPUT_PATH = f"annotated_image_{step}.jpg"
    detected_tags = detect_apriltags(image=img_array, output_path=OUTPUT_PATH)
    if not detected_tags:
        # No tags detected: rotate based on last known tag side.
        if LAST_TAG_SIDE == "left":
            print("WARNING: No tags detected; last seen on left. Rotating left to search for tags.")
            left_motor.setVelocity(TURN_SPEED_RATIO * ROTATION_SPEED)
            right_motor.setVelocity(ROTATION_SPEED)
        else:
            print("WARNING: No tags detected; last seen on right. Rotating right to search for tags.")
            left_motor.setVelocity(ROTATION_SPEED)
            right_motor.setVelocity(TURN_SPEED_RATIO * ROTATION_SPEED)
        return

    # 3) Estimate the robot pose (x, y, theta) in mm and radians.
    pose = estimate_robot_pose(detected_tags)
    if pose is None:
        print("ERROR: Could not estimate robot pose from tag detections.")
        left_motor.setVelocity(0)
        right_motor.setVelocity(0)
        return
    robot_x, robot_y, robot_theta = pose
    print(f"Robot position: x = {robot_x:.1f} mm, y = {robot_y:.1f} mm, heading = {math.degrees(robot_theta):.1f}°")
    
    # Adjust heading if necessary.
    robot_theta = (robot_theta + math.pi) % (2 * math.pi)
    if robot_theta > math.pi:
        robot_theta -= 2 * math.pi
    print(f"Robot pose: x={robot_x:.1f} mm, y={robot_y:.1f} mm, heading={math.degrees(robot_theta):.1f}°")


    # 4) Get the home (corner) coordinates from destination IDs.
    home_x, home_y = get_destination_coordinate()
    print(f"Home destination coordinates: x = {home_x:.1f} mm, y = {home_y:.1f} mm")

    # 5) Compute the distance and heading error.
    dx = home_x - robot_x
    dy = home_y - robot_y
    distance_to_home = math.hypot(dx, dy)
    print(f"Distance to home: {distance_to_home:.1f} mm")
    
    # If within threshold, execute the arrival routine.
    if distance_to_home < DISTANCE_THRESHOLD:
        arrival_routine(left_motor, right_motor, robot)
        return

    # Compute heading error: the difference between desired and current heading.
    target_angle = math.atan2(dy, dx)
    angle_diff = (target_angle - robot_theta + math.pi) % (2 * math.pi) - math.pi

    # Determine which side the home position is on relative to the robot's current heading.
    if angle_diff > 0:
        LAST_TAG_SIDE = "left"
        print("TARGET DIRECTION: Home is to the LEFT.")
    else:
        LAST_TAG_SIDE = "right"
        print("TARGET DIRECTION: Home is to the RIGHT.")

    print(f"Heading details: target angle = {math.degrees(target_angle):.1f}°, current heading = {math.degrees(robot_theta):.1f}°, angle difference = {math.degrees(angle_diff):.1f}°.")

    # 6) Decide whether to rotate in place or drive forward with turning.
    angle_threshold = math.radians(30)  # e.g., 30° threshold for switching behaviors.
    if abs(angle_diff) > angle_threshold:
        print("Angle diff is large; rotating in place to face home.")
        if angle_diff > 0:
            left_speed = -ROTATION_SPEED
            right_speed = ROTATION_SPEED
        else:
            left_speed = ROTATION_SPEED
            right_speed = -ROTATION_SPEED
    else:
        print("Angle diff is small; driving forward while turning toward home.")
        base_speed = min(FORWARD_SPEED, MAX_MOTOR_SPEED)
        turn = ANGLE_GAIN * angle_diff
        left_speed = base_speed - turn
        right_speed = base_speed + turn
        left_speed = max(-MAX_MOTOR_SPEED, min(MAX_MOTOR_SPEED, left_speed))
        right_speed = max(-MAX_MOTOR_SPEED, min(MAX_MOTOR_SPEED, right_speed))

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)
    print(f"Motor commands: left={left_speed:.2f}, right={right_speed:.2f}, angle diff={math.degrees(angle_diff):.1f}°, distance to home={distance_to_home:.1f} mm.")


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
    global FX, FY, TAG_SIDE_METERS
    
    # Enhance the image to improve detection accuracy
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

    cx = image.shape[1] / 2 # principal point x
    cy = image.shape[0] / 2 # principal point y
    tags = detector.detect(enhanced_gray, estimate_tag_pose=True, camera_params=(FX, FY, cx, cy), tag_size=TAG_SIDE_METERS)

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

        # ----------------------------
        # Drawing / annotation 
        # (unchanged from your version)
        # ----------------------------
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

    if output_path:
        cv2.imwrite(output_path, image)
        print(f"\nAnnotated image saved to '{output_path}'.")

    # Sort by X-center just like you had
    sorted_tags = sorted(detected_tags, key=lambda t: t['center'][0])
    print("Detected Tags Ordered by X-Center Position:")
    for t in sorted_tags:
        print(f"ID: {t['id']}, Center=({t['center'][0]:.2f},{t['center'][1]:.2f}), "
              f"Dist={t['pose']['distance']:.1f}mm, Bearing={t['pose']['bearing']:.2f} rad")

    return sorted_tags


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

def parse_boxes_into_detections(boxes, step_count):
    """
    Turn a YOLOv11 Results.boxes object into your `detections` list.
    """
    detections = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x_center = int(((x1 + x2) / 2).item())
        class_index = int(box.cls.item()) if hasattr(box, "cls") else -1
        confidence  = float(box.conf.item()) if hasattr(box, "conf") else 0.0
        track_id    = int(box.id.item())  if hasattr(box, "id")   else -1

        detected_class = (
            OBJECT_DETECTION_CLASSES[class_index]
            if 0 <= class_index < len(OBJECT_DETECTION_CLASSES)
            else "Unknown"
        )

        print(f"[ID {track_id}] {detected_class} conf={confidence:.2f} bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
        detections.append({
            "id":         track_id,
            "class":      detected_class,
            "confidence": confidence,
            "center_x":   x_center,
            "bbox":       (float(x1), float(y1), float(x2), float(y2))
        })
    return detections


def main():
    global CHASE_BALL, RETURN_HOME, FORWARD_SPEED, ROTATION_SPEED, TURN_RATIO, COMPETITION, COLLECT_DATA, GO_HOME_TIMER, IMAGE_WIDTH, IMAGE_HEIGHT, FRAMES_TILL_TARGET_SWITCH, DETECTION_FRAME_INTERVAL, MODEL_PATH, CAMERA_NAME, OBJECT_DETECTION_CLASSES, COMPETITION_START_TIME, HOME_IDS, TAG_POSITIONS, HOME_POSITIONS
    current_target_id  = None
    frames_since_seen  = 0
    model = load_model(MODEL_PATH)
    
    if COMPETITION:
        print(f"Competition mode enabled - waiting for {COMPETITION_START_TIME} seconds.")
    
    robot = Robot()
    timestep, camera, left_motor, right_motor = init_environment(robot)
    print(f"Camera resolution: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    step_count = 0
    prev_x_positions = []

    if COMPETITION:
        time.sleep(COMPETITION_START_TIME)
        print(f"{COMPETITION_START_TIME} seconds have passed. Starting the competition.")

    # Initialize a timer for BALL_CHASE mode
    chase_start_time = time.time()
    # Keep track of the previous mode to detect mode transitions
    previous_mode = "CHASE_BALL"
    
    while robot.step(timestep) != -1:
        step_count += 1

        # Capture the image from the camera
        img = camera.getImage()
                
        if COLLECT_DATA and step_count % DATA_COLLECTION_INTERVAL == 0:
            pil_img = Image.frombytes('RGBA', (IMAGE_WIDTH, IMAGE_HEIGHT), img, 'raw', 'BGRA')
            pil_img.save(f"frames/raw_frame_{step_count}.png")
            print(f"Frame {step_count} saved for data collection.")
        
        # Check if 15 seconds have elapsed in BALL_CHASE mode.
        if CHASE_BALL and (time.time() - chase_start_time >= GO_HOME_TIMER):
            CHASE_BALL = False
            RETURN_HOME = True
            print(f"{GO_HOME_TIMER} seconds elapsed in BALL_CHASE mode. Switching to RETURN_HOME mode.")
        
        if CHASE_BALL:
            # only do detection every N frames 
            if step_count % DETECTION_FRAME_INTERVAL == 0:
                detections = ball_detection(img, camera, model, step_count)

                if detections:
                    frames_since_seen = 0
                    seen_ids = [d['id'] for d in detections]
                    if current_target_id not in seen_ids or frames_since_seen >= FRAMES_TILL_TARGET_SWITCH or current_target_id is None:
                        # choose the ball with the *lowest* center-y pixel (closest)
                        best = max(detections, key=lambda d: ((d['bbox'][1] + d['bbox'][3]) / 2))
                        current_target_id = best['id']
                        print(f"Switched target → ID {current_target_id}")
                    target = next(d for d in detections if d['id'] == current_target_id)

                    # Compute horizontal error and P‐turn
                    x_err = target['center_x'] - IMAGE_WIDTH/2
                    Kp    = 0.005
                    turn  = Kp * x_err
                 
                    # drive forward + turn toward the ball (swap signs)
                    left_motor .setVelocity(FORWARD_SPEED + turn)
                    right_motor.setVelocity(FORWARD_SPEED - turn)

                else:
                    # no detections at all this frame
                    frames_since_seen += 1

                    # if we’ve now missed the target for too long, reset it
                    if frames_since_seen >= FRAMES_TILL_TARGET_SWITCH:
                        print(f"Lost target ID {current_target_id}, will reselect.")
                        current_target_id = None

                    # optional: spin in place to look for balls
                    left_motor.setVelocity (ROTATION_SPEED)
                    right_motor.setVelocity(-ROTATION_SPEED)

        elif RETURN_HOME:
            if step_count % DETECTION_FRAME_INTERVAL == 0:
                return_home_deterministic(img, camera, left_motor, right_motor, robot, step=step_count // DETECTION_FRAME_INTERVAL)
            if CHASE_BALL:
                chase_start_time = time.time()
                print("Returned home. Switching back to BALL_CHASE mode and resetting timer.")    
    robot.cleanup()

if __name__ == "__main__":
    main()