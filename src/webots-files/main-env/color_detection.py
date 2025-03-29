import cv2
import numpy as np
import os

# Parameters
IMAGE_NAME = 'arena_1.jpeg'
H_COLOR_THRESHOLD = 0.6 # this parameter specifies the horizental percentage of the image to conside the walls


# Define HSV color ranges
color_ranges = {
    "Green": ([40, 40, 40], [80, 255, 255]),    
    "Purple": ([125, 40, 40], [160, 255, 255]),  
    "Yellow": ([20, 100, 100], [30, 255, 255]),  
    "Orange": ([10, 100, 100], [20, 255, 255])   
}

# Load the image with correct file path handling
current_dir = os.path.dirname(__file__)
relative_path = os.path.join(current_dir, IMAGE_NAME)
IMAGE_PATH = os.path.abspath(relative_path)
image = cv2.imread(IMAGE_PATH)

if image is None:
    print(f"Error: Could not read image from {IMAGE_PATH}")
    exit()

# Resize image (optional)
image = cv2.resize(image, (600, 400))

# Get image height to define the lower 30% region
height, width, _ = image.shape
lower_bound = int(height * (1 - H_COLOR_THRESHOLD))

# Convert image to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Process each color
for color, (lower, upper) in color_ranges.items():
    lower = np.array(lower, dtype=np.uint8)
    upper = np.array(upper, dtype=np.uint8)

    # Create mask for the color
    mask = cv2.inRange(hsv, lower, upper)

    # Find contours of the detected color regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Ignore small detections
            x, y, w, h = cv2.boundingRect(contour)

            # Check if the bounding box is in the lower 30% of the image
            if y + h >= lower_bound:
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 2)
                cv2.putText(image, color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# Draw a horizontal line to indicate the lower 30% region
cv2.line(image, (0, lower_bound), (width, lower_bound), (0, 255, 255), 2)

# Show output
cv2.imshow("Color Detection (Lower 30%)", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
