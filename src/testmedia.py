import os
import cv2
import platform
import subprocess
from ultralytics import YOLO

def open_folder(path):
    """Safely opens the folder in File Explorer."""
    if os.path.exists(path):
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])

def process_frame(frame, model):
    """Applies custom color-coded bounding boxes to a single frame."""
    results = model(frame, verbose=False)
    for r in results:
        for box in r.boxes:
            # Get coordinates, confidence, and class
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = round(float(box.conf[0]) * 100, 1)
            cls = int(box.cls[0])
            class_name = model.names[cls]

            # Custom Color Logic
            if class_name == "Helmet":
                color = (0, 255, 0)      # Green
            elif class_name == "No-Helmet":
                color = (0, 0, 255)      # Red
            else:
                color = (255, 0, 0)      # Blue (person)

            # Draw Box and Label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            label = f'{class_name} {conf}%'
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - t_size[1] - 10), (x1 + t_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 7), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame

def main():
    model_path = 'runs/detect/helmet_yolo_model/weights/best.pt'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    model = YOLO(model_path)
    output_dir = os.path.join('media_test', 'detections')
    os.makedirs(output_dir, exist_ok=True)

    print("\n--- YOLO Helmet Detection Media Tester ---")
    source_path = input("Enter the path to your image or video: ").strip().replace('"', '').replace("'", "")
    
    if not os.path.exists(source_path):
        print("Error: File not found.")
        return

    file_name = os.path.basename(source_path)
    save_path = os.path.join(output_dir, file_name)
    is_video = source_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))

    if not is_video:
        # Process Image
        img = cv2.imread(source_path)
        processed_img = process_frame(img, model)
        cv2.imwrite(save_path, processed_img)
    else:
        # Process Video
        cap = cv2.VideoCapture(source_path)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        print(f"Processing video frames...")
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            processed_frame = process_frame(frame, model)
            out.write(processed_frame)
        
        cap.release()
        out.release()

    print("\n" + "="*40)
    print(f"SUCCESS! Processed file: {save_path}")
    open_folder(os.path.abspath(output_dir))
    print("="*40)

if __name__ == "__main__":
    main()