import torch
import os
import sys
from ultralytics import YOLO

current_dir = os.path.dirname(__file__)
relative_path = os.path.join(current_dir, '..', '..', '..', '..', '..', 'weights', 'real-world-detector.pt')
MODEL_PATH = os.path.abspath(relative_path)

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

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

# Assume IMAGE_HEIGHT and IMAGE_WIDTH are defined (e.g., 480, 640)
dummy_input = torch.randn(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)

model = load_model(MODEL_PATH)

torch.onnx.export(
    model, 
    dummy_input, 
    "real-world-detector.onnx", 
    opset_version=11,
    input_names=["input"],
    output_names=["output"],
    export_params=True
)
