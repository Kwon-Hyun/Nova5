
import cv2
from pyzbar.pyzbar import decode
import numpy as np 

def detect_qr_from_camera():
    # Open the default camera (0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video camera.")
        return

    print("Press 'q' to exit the QR code detection.")

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Detect QR codes in the frame
        decoded_objects = decode(frame)
        
        # Process each detected QR code
        for obj in decoded_objects:
            # Extract bounding box for the QR code
            points = obj.polygon
            if len(points) == 4:
                pts = [(point.x, point.y) for point in points]
                # Draw a rectangle around the detected QR code
                cv2.polylines(frame, [np.array(pts, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
            
            # Extract and print QR code data
            qr_data = obj.data.decode("utf-8")
            print("QR Code detected:", qr_data)

            # Display the QR data on the screen
            cv2.putText(frame, qr_data, (pts[0][0], pts[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the frame
        cv2.imshow("QR Code Detector", frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Run the QR code detector from camera
detect_qr_from_camera()
