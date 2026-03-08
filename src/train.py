import torch
from ultralytics import YOLO

def main():
    # 1. Automatically detect if a GPU is available
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Starting training on device: {device}")

    # 2. Load the base YOLOv8 Nano model
    model = YOLO('yolov8n.pt')

    # 3. Train the model on your custom dataset
    # Make sure the path to data.yaml is correct relative to where you run the script!
    results = model.train(
        data='datasets/data.yaml',  
        epochs=30,                  # Number of times the model will look at the entire dataset
        imgsz=640,                  # Standard image size for YOLO
        batch=16,                   # Number of images processed at once
        device=device,              # Automatically uses GPU if available
        name='helmet_yolo_model'    # Name of the folder where results are saved
    )

    print("Training complete! Check the 'runs/detect/helmet_yolo_model' folder for your best weights.")

if __name__ == '__main__':
    main()