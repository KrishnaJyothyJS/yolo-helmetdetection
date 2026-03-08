import cv2
from ultralytics import YOLO
import os

def test_media(source_path):
    # 1. Load your custom model
    model = YOLO('runs/detect/helmet_yolo_model/weights/best.pt')

    # 2. Run Inference
    # This automatically detects if it is an image or video
    results = model(source_path, save=True, project="runs/test_results", name="media_test")

    print(f"Results saved to: runs/test_results/media_test")

if __name__ == "__main__":
    # Change this to the path of the file you want to test!
    # Example: 'test_assets/biker_video.mp4' or 'test_assets/group_photo.jpg'
    path_to_test = input("Enter the path to your image or video file: ")
    
    if os.path.exists(path_to_test):
        test_media(path_to_test)
    else:
        print("Error: File not found. Check the path!")