import cv2
import os
import numpy as np
import ncnn
import torch
import time  # Import time module

# Configuration parameters
CONFIDENCE_THRESHOLD = 0.5
COLLECT_INFERENCE_DATA = True  # Set to True if you want to save the predicted images
step_count = 0

# Function to run NCNN inference
def test_inference(frame):
    # Convert frame to a tensor with shape (1, 3, 640, 640)
    frame_resized = cv2.resize(frame, (640, 640))  # Resize the frame
    frame_tensor = torch.from_numpy(frame_resized).float().permute(2, 0, 1).unsqueeze(0)  # CHW format for input

    out = []

    with ncnn.Net() as net:
        # Load the NCNN model (ensure you have the correct paths for .param and .bin)
        net.load_param("/home/unibots/unibots/weights/new-weights/0-simple-NCNN/model.ncnn.param")
        net.load_model("/home/unibots/unibots/weights/new-weights/0-simple-NCNN/model.ncnn.bin")

        with net.create_extractor() as ex:
            # Input NCNN data
            ex.input("in0", ncnn.Mat(frame_tensor.squeeze(0).numpy()).clone())

            # Extract the output
            _, out0 = ex.extract("out0")
            out.append(torch.from_numpy(np.array(out0)).unsqueeze(0))

    if len(out) == 1:
        return out[0]
    else:
        return tuple(out)

# Open the webcam (device 0)
cap = cv2.VideoCapture(0)
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

    # Measure the time taken for inference
    start_time = time.time()  # Record the start time
    
    # Run NCNN inference (converted model)
    ncnn_output = test_inference(frame)  # Get inference results from NCNN model
    
    end_time = time.time()  # Record the end time
    inference_time = end_time - start_time  # Calculate the time taken for inference
    
    print(f"Inference time: {inference_time:.4f} seconds")  # Print the time taken for inference

    # Process NCNN output to extract bounding box coordinates and calculate x positions
    # Assuming ncnn_output contains bounding boxes in a usable format (e.g., [x1, y1, x2, y2])
    # Note: You'll need to adjust the following based on the actual structure of `ncnn_output`
    if ncnn_output is not None:
        print(f"NCNN Output (shape): {ncnn_output.shape}")  # Print the shape of the output tensor
        print(f"NCNN Output (raw values): {ncnn_output}")  # Print the raw values in the output tensor

        # Check how many items are in the output tensor and extract them accordingly
        for box in ncnn_output[0]:  # Iterate through detected boxes
            print(f"Box details: {box}")  # Print out the box to inspect its structure
            try:
                # Try to extract coordinates if possible
                if len(box) >= 4:
                    # Extract the first four values as box coordinates
                    x1, y1, x2, y2 = box[:4].tolist()
                    # Calculate the center x position
                    x_center = (x1 + x2) / 2  
                    x_center_int = int(x_center)
                    print(f"Center x coordinate (NCNN): {x_center_int}")  # Print the x center position
                else:
                    print(f"Skipping box with unexpected number of values: {len(box)}")
            except Exception as e:
                print(f"Error processing box: {e}")
    else:
        print(f"Image {step_count}: No objects detected.")

    step_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()