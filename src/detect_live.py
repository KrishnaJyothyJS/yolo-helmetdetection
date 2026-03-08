import cv2
import time
from ultralytics import YOLO

def main():
    # 1. Load your CUSTOM trained model weights
    # Once your teammate finishes training, the file will be in:
    # 'runs/detect/helmet_yolo_model/weights/best.pt'
    # For now, we will use 'yolov8n.pt' so the code doesn't crash before training.
    try:
        model = YOLO('runs/detect/helmet_yolo_model/weights/best.pt')
        print("Using Custom Trained Model!")
    except:
        model = YOLO('yolov8n.pt')
        print("Custom weights not found. Using default YOLOv8n for testing.")

    # 2. Open Webcam
    cap = cv2.VideoCapture(0)
    
    # Optional: Set resolution lower for better FPS on the IdeaPad
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_time = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 3. Run YOLO inference
        results = model(frame, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Get confidence and class index
                conf = round(float(box.conf[0]) * 100, 1)
                cls = int(box.cls[0])
                
                # Get class name from the model
                class_name = model.names[cls]

                # 4. Color Logic based on your data.yaml
                # Helmet = Green, No-Helmet = Red, person = Blue
                if class_name == "Helmet":
                    color = (0, 255, 0)      # Green
                elif class_name == "No-Helmet":
                    color = (0, 0, 255)      # Red
                else:
                    color = (255, 0, 0)      # Blue (for 'person')

                # Draw Bounding Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                # Draw Label Background
                label = f'{class_name} {conf}%'
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (x1, y1 - t_size[1] - 10), (x1 + t_size[0], y1), color, -1)
                
                # Put Text
                cv2.putText(frame, label, (x1, y1 - 7), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 5. Calculate and Display FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f'FPS: {int(fps)}', (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # 6. Show the frame
        cv2.imshow('Helmet Detection Competition Demo', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()