import cv2
import os

output_dir = 'captured_frames'
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

frame_counter = 0
print("Capturing frames. Press Ctrl+C in the console to stop capturing.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        print("Frame shape:", frame.shape)
        # Construct filename and save frame to disk
        filename = os.path.join(output_dir, f'frame_{frame_counter:05d}.jpg')
        cv2.imwrite(filename, frame)
        print(f"Captured image saved as {filename}")
        frame_counter += 1
except KeyboardInterrupt:
    print("Capture stopped by user.")
finally:
    cap.release()
    print("Capture ended. All frames saved in:", output_dir)