import cv2
from ultralytics import YOLO

def main():
    # 1. Load the pre-trained YOLOv8 Nano model (it will download automatically the first time)
    print("Loading YOLOv8 model... This might take a few seconds.")
    model = YOLO('yolov8n.pt') 
    
    # 2. Open the webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("AI Vision Active! Press 'q' to quit.")

    # Start the video loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # 3. Run YOLO inference on the current frame
        # We use stream=True because it is optimized for live video
        results = model(frame, stream=True)

        # 4. Draw the bounding boxes on the frame
        for r in results:
            # The .plot() method automatically draws the boxes and labels
            annotated_frame = r.plot() 

        # 5. Display the AI-annotated frame
        cv2.imshow('YOLOv8 Live AI Feed', annotated_frame)

        # Quit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Shutting down AI feed...")
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()