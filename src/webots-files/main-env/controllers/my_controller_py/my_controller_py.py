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
relative_path = os.path.join(current_dir, '..', '..', 'weights', 'best.pt')
MODEL_PATH = os.path.abspath(relative_path)

################
### SETTINGS ###
################

# MODEL_PATH = r"yolo11s.pt" # Use standard model instead    
CONFIDENCE_THRESHOLD = 0.15
DETECTION_FRAME_INTERVAL = 25 # controls how many frames are skipped between apriltag / ball detection is performed
CAMERA_NAME = "camera"
DISTANCE_THRESHOLD = 250 # 300.0
HOME_IDS = [0, 23]
FORWARD_SPEED = 6.0  # Adjust this value as needed
ROTATION_SPEED = 4.0
MAX_MOTOR_SPEED = 6.28 # WeBots speed limit:= 6.28 rad/s
ANGLE_GAIN = 3
TURN_RATIO = 0.7

### CAMERA PARAMETERS ###
FX = 1600 # focal length along x-axis
FY = 1600 # focal length along y-axis
TAG_SIDE_METERS = 0.1 # example: 10cm wide tags

# ROBOT STATES
CHASE_BALL = False
RETURN_HOME = True

# INDEPENDENT STATES
COLLECT_DATA = True # save frames to disk, to create training data

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
    try:
        width = camera.getWidth()
        height = camera.getHeight()
        # Convert the raw image data to a NumPy array
        img_array = np.frombuffer(img_bytes, dtype=np.uint8).reshape((height, width, 4))
        # Convert RGBA to RGB by removing the alpha channel and make a copy to ensure writeability
        img_rgb = img_array[:, :, :3].copy()
        return img_rgb
    except Exception as e:
        print(f"Error converting bytes to NumPy array: {e}")
        return None


def ball_detection(img, camera, model):
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
        image_np = np.array(image)
        print("Model inference starting...")

        # Run the YOLOv11 model on the image
        results = model(image_np, conf=CONFIDENCE_THRESHOLD, save=True)

        # Process and print detected objects
        result = results[0]  # Since there's only one image
        boxes = result.boxes  # Boxes object for bounding box outputs
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


def return_home(img_bytes, camera, left_motor, right_motor, robot, step, destination_ids=[0, 23]):
    global CHASE_BALL
    global RETURN_HOME
    global DISTANCE_THRESHOLD
    global FORWARD_SPEED
    global ROTATION_SPEED
    global MAX_MOTOR_SPEED
    global ANGLE_GAIN

    # 1) Convert bytes → NumPy array
    img_array = bytes_to_numpy(img_bytes, camera)
    if img_array is None:
        print("Failed to convert image bytes to NumPy array.")
        return

    # 2) Detect AprilTags in the image
    OUTPUT_PATH = f"annotated_image_{step}.jpg"
    detected_tags = detect_apriltags(image=img_array, output_path=OUTPUT_PATH)
    if not detected_tags:
        print("No tags detected; rotating in place to find tags.")
        # Slowly rotate in place until the next detection
        left_motor.setVelocity(-ROTATION_SPEED)
        right_motor.setVelocity(ROTATION_SPEED)
        return

    # 3) Estimate the robot pose (x, y, theta) in mm + radians
    pose = estimate_robot_pose(detected_tags)
    if pose is None:
        print("Could not estimate robot pose from detections.")
        left_motor.setVelocity(0)
        right_motor.setVelocity(0)
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

    if distance_to_home < DISTANCE_THRESHOLD:
        # 1) Move backwards for 1 second
        print(f"Within {DISTANCE_THRESHOLD}mm threshold. Moving backwards...")
        left_motor.setVelocity(-FORWARD_SPEED)
        right_motor.setVelocity(-FORWARD_SPEED)
        start = time.time()
        while (time.time() - start) < 1.0:  # 1 second backward
            if robot.step(int(robot.getBasicTimeStep())) == -1:
                break

        # 2) Rotate away from the home position for 1 second
        print("Rotating away from home corner...")
        left_motor.setVelocity(FORWARD_SPEED)
        right_motor.setVelocity(-FORWARD_SPEED)
        start = time.time()
        while (time.time() - start) < 1.0:  # 1 second rotate
            if robot.step(int(robot.getBasicTimeStep())) == -1:
                break

        # 3) Switch back to CHASE_BALL mode
        left_motor.setVelocity(0)
        right_motor.setVelocity(0)
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
    left_speed = base_speed + turn
    right_speed = base_speed - turn

    left_speed = max(-MAX_MOTOR_SPEED, min(MAX_MOTOR_SPEED, left_speed))
    right_speed = max(-MAX_MOTOR_SPEED, min(MAX_MOTOR_SPEED, right_speed))

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)

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
    global FX, FY, TAG_SIDE_METERS
    
    # Enhance the image to improve detection accuracy
    print(f"Image type before enhancement: {type(image)}")
    print(f"Image writeable before enhancement: {image.flags.writeable}")
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
        # For heading, you could do a more sophisticated approach
        # or just pick the first detection’s yaw as an estimate:
        theta_est = detections[0]['pose'].get('yaw', 0.0)
        return (x_est, y_est, theta_est)
    else:
        return None


def main():
    # === Configuration ===
    # Path to the YOLOv11 model weights
    global CHASE_BALL
    global RETURN_HOME
    global FORWARD_SPEED
    global ROTATION_SPEED
    global TURN_RATIO
    model = load_model(MODEL_PATH)
    # === Initialize Robot ===
    robot = Robot()
    timestep, camera, left_motor, right_motor = init_environment(robot)

    print("Controller initialized. Robot is moving forward and capturing images.")

    # Initialize detection counter
    step_count = 0
    prev_x_positions = []
    # Main loop: keep running until the simulation is stopped
    while robot.step(timestep) != -1:
        step_count += 1

        # Capture the image
        img = camera.getImage()
        
        if COLLECT_DATA:
            pil_img = Image.frombytes('RGB', (camera.width, camera.height), img)
            filename = f"img_{step_count}.png"
            pil_img.save(filename)

        
        if CHASE_BALL:
            x_positions = []

            if img:
                # Perform detection at specified intervals to manage computational load
                if step_count % DETECTION_FRAME_INTERVAL == 0:
                    x_center_int = ball_detection(img, camera, model)
                    if x_center_int is not None:
                        x_positions.append(x_center_int)

            # check if x_positions is empty, only overwrite if not empty or last frame was also empty
            if step_count % DETECTION_FRAME_INTERVAL == 0:
                if not x_positions and prev_x_positions:
                    a = x_positions.copy()
                    x_positions = prev_x_positions.copy()
                    prev_x_positions = a.copy()

            # move robot according to path planning (using x_positions, every frame)
            if x_positions:
                if x_positions[-1] > camera.getWidth() / 2:
                    # move right
                    print("move to the right")
                    left_motor.setVelocity(FORWARD_SPEED)
                    right_motor.setVelocity(FORWARD_SPEED * TURN_RATIO)
                else:
                    print("move to the left")
                    left_motor.setVelocity(FORWARD_SPEED * TURN_RATIO)
                    right_motor.setVelocity(FORWARD_SPEED)

            # Optional: Add a sleep or control the loop frequency if needed
            # time.sleep(0.01)
        elif RETURN_HOME:
            global HOME_IDS
            if step_count % DETECTION_FRAME_INTERVAL == 0:
                return_home(img, camera, left_motor, right_motor, robot, step=step_count//DETECTION_FRAME_INTERVAL, destination_ids=HOME_IDS)       
    # Cleanup (optional in Python)
    robot.cleanup()


if __name__ == "__main__":
    main()