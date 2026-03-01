import cv2

def main():
    # 0 is usually the default ID for your laptop's built-in webcam. 
    # If you plug in an external USB camera later, it might be 1 or 2.
    cap = cv2.VideoCapture(0)

    # Check if the webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        return

    print("Webcam successfully opened! Press 'q' to quit the feed.")

    # Start a loop to continuously grab frames from the webcam
    while True:
        # Read a single frame
        ret, frame = cap.read()

        # If the frame was not grabbed correctly, break the loop
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Display the frame in a window named 'Live Demo - Testing'
        cv2.imshow('Live Demo - Testing', frame)

        # Wait for 1 millisecond for a key press. 
        # If the key pressed is 'q', break the loop and close the window.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Closing the webcam feed...")
            break

    # Once the loop breaks, release the webcam hardware and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()