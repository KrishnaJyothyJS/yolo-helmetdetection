import cv2
import os
from ultralytics import YOLO

def process_frame(frame, model):
    """Applies YOLO detection and custom coloring to a single frame."""
    results = model(frame, conf=0.5)  # Setting confidence to 50%
    
    for r in results:
        for box in r.boxes:
            # Get coordinates and class info
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = round(float(box.conf[0]) * 100, 1)
            class_name = model.names[cls]

            # Custom Color Logic
            if class_name == "Helmet":
                color = (0, 255, 0)      # Green
            elif class_name == "No-Helmet":
                color = (0, 0, 255)      # Red
            else:
                color = (255, 0, 0)      # Blue (Person)

            # Draw Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw Label
            label = f'{class_name} {conf}%'
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - t_size[1] - 10), (x1 + t_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 7), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame

def main():
    # Load your custom model
    model_path = 'runs/detect/helmet_yolo_model/weights/best.pt'
    if not os.path.exists(model_path):
        print("Error: best.pt not found! Use yolov8n.pt for testing.")
        model = YOLO('yolov8n.pt')
    else:
        model = YOLO(model_path)

    path = input("Enter path to image or video (e.g., test_assets/test.jpg): ").strip()
    
    if not os.path.exists(path):
        print("File not found!")
        return

    # Check if file is an image or video
    extension = os.path.splitext(path)[1].lower()
    
    if extension in ['.jpg', '.jpeg', '.png']:
        # IMAGE MODE
        image = cv2.imread(path)
        processed_image = process_frame(image, model)
        cv2.imshow('Image Detection Test', processed_image)
        print("Press any key to close the image.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif extension in ['.mp4', '.avi', '.mov']:
        # VIDEO MODE
        cap = cv2.VideoCapture(path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame = process_frame(frame, model)
            cv2.imshow('Video Detection Test', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unsupported file format!")

if __name__ == "__main__":
    main()