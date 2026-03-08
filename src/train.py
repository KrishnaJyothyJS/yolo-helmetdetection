import torch
from ultralytics import YOLO

def main():
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Resuming training on device: {device}")

    # 1. Point to the 'last.pt' checkpoint instead of the base model
    # This allows the model to pick up right where it crashed!
    model = YOLO('runs/detect/helmet_yolo_model/weights/last.pt')

    results = model.train(
        data='datasets/data.yaml',
        epochs=30,
        imgsz=640,
        batch=8,           # Reduced slightly to be safe with GPU memory
        device=device,
        workers=0,         # CRITICAL: Fixes the 'resource already mapped' error on Windows
        pin_memory=False,  # CRITICAL: Fixes the memory thread crash
        resume=True,       
        name='helmet_yolo_model',
        exist_ok=True      # Overwrites the existing folder instead of creating a new one
    )

if __name__ == '__main__':
    main()