import torch
from ultralytics import YOLO

def main():
    print("Resuming training...")

    # 1. Point to the checkpoint where it crashed
    model = YOLO('runs/detect/helmet_yolo_model/weights/last.pt')

    # 2. Resume training and apply the Windows crash fix (workers=0)
    # YOLO automatically remembers your data.yaml, batch size, and the 30 epochs limit!
    results = model.train(
        resume=True,
        workers=0
    )

if __name__ == '__main__':
    main()