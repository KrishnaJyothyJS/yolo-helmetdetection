import os
import subprocess
import platform
from ultralytics import YOLO

def open_folder(path):
    """Automatically opens the folder in File Explorer based on the OS."""
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":  # macOS
        subprocess.Popen(["open", path])
    else:  # Linux
        subprocess.Popen(["xdg-open", path])

def main():
    # 1. Path to your custom-trained 'brain'
    model_path = 'runs/detect/helmet_yolo_model/weights/best.pt'
    
    if not os.path.exists(model_path):
        print(f"Error: Custom model not found at {model_path}")
        print("Make sure your teammate has finished training and the file is in the right place!")
        return

    # 2. Load the model
    model = YOLO(model_path)

    # 3. Get the input file path from the user
    print("\n--- YOLO Helmet Detection Media Tester ---")
    source_path = input("Enter the path to your image or video (e.g., test_assets/test.mp4): ").strip()
    
    if not os.path.exists(source_path):
        print(f"Error: The file '{source_path}' does not exist.")
        return

    # 4. Run Inference and Save Results
    # project='media_test' creates the main folder
    # name='detections' creates the sub-folder
    # exist_ok=True prevents it from creating 'detections2', 'detections3', etc.
    print(f"\nProcessing: {os.path.basename(source_path)}...")
    
    results = model.predict(
        source=source_path,
        save=True,           # Saves the file with boxes and labels
        project='media_test', 
        name='detections',    
        exist_ok=True,       
        conf=0.5,            # Only show boxes with >50% confidence
        line_width=2         # Adjusts thickness of the bounding box
    )

    # 5. Final Output
    output_dir = os.path.join('media_test', 'detections')
    print("\n" + "="*40)
    print(f"SUCCESS! Processed file is in: {output_dir}")
    print("Opening the results folder now...")
    print("="*40)

    # Automatically open the folder for the demo
    open_folder(os.path.abspath(output_dir))

if __name__ == "__main__":
    main()