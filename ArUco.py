import numpy as np
import cv2
import pyrealsense2 as rs

class RealSenseArucoDetector:
    def __init__(self, aruco_type="DICT_5X5_250", marker_size=0.05):
        # Define ArUco dictionary and parameters
        self.ARUCO_DICT = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
            "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
            "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
            "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
            "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
            "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
            "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
            "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
            "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
            "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
            "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
            "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
            "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
            "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
            "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
            "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
            "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
            "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
            "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
        }
        self.arucoDict = cv2.aruco.getPredefinedDictionary(self.ARUCO_DICT[aruco_type])
        self.arucoParams = cv2.aruco.DetectorParameters_create()
        self.marker_size = marker_size

        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Start streaming
        self.pipeline.start(self.config)

        # Get camera intrinsic parameters for pose estimation
        profile = self.pipeline.get_active_profile()
        color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.cameraMatrix = np.array([[color_intrinsics.fx, 0, color_intrinsics.ppx],
                                      [0, color_intrinsics.fy, color_intrinsics.ppy],
                                      [0, 0, 1]])
        self.distCoeffs = np.zeros((4, 1))  # Assuming no lens distortion

    def aruco_display(self, corners, ids, image, rvecs, tvecs, w, h, width, height, depth_frame):
        if len(corners) > 0:
            ids = ids.flatten()
            for i, (markerCorner, markerID) in enumerate(zip(corners, ids)):
                corners = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners
                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))

                # Draw lines on the marker
                cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
                cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
                cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
                cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

                # Calculate center point
                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)

                # Calculate rotation angle
                # Use the top edge (topLeft to topRight) to determine the angle
                angle_rad = np.arctan2(topRight[1] - topLeft[1], topRight[0] - topLeft[0])
                angle_deg = np.degrees(angle_rad)

                # change into UR rotation UR equation pi 3.14159 * ( angle_deg/180 )
                ur_degree = 3.14159 * (angle_deg/180)

                #Calculate distance
                # Map the (x, y) coordinates back to the original image size
                x_original = int(cX * (w / width))
                y_original = int(cY * (h / height))

                # Get the distance from the depth frame
                distance = depth_frame.get_distance(x_original, y_original)


                # Get pose estimation (x, y, z) for the marker
                rvec, tvec = rvecs[i][0], tvecs[i][0]

                # Display the angle on the image
                label = f"ID: {markerID} Angle: {angle_deg:.2f}"
                cv2.putText(image, label, (topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)
                
                # Display UR degree & distance
                # label_degree = f"URDegree: {ur_degree:.2f} dis: {distance:.2f}"
                label_degree = f"Distance: {distance:.2f}"
                cv2.putText(image, label_degree, (topLeft[0], topLeft[1] + 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 0, 0), 2)


                pos_label = f"X: {tvec[0]:.2f} Y: {tvec[1]:.2f} Z: {tvec[2]:.2f}"
                cv2.putText(image, pos_label, (topLeft[0], topLeft[1] + 60), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 0, 0), 2)

                # Draw the axes on the marker
                #cv2.drawFrameAxes(image, cameraMatrix, distCoeffs, rvec, tvec, 0.05)


                #calculate tray & magzine position by reference QR code
                # carrier_pos = f"X tray: {Xres:.2f} Y: {Yres:.2f}"
                # cv2.putText(image, carrier_pos, (topLeft[0], topLeft[1] + 90), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)



            return image, ur_degree, distance
        else:
            print("[Inference] No ArUco markers detected.")
            return image, None, None

    def get_marker_data(self):
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame:
            return None, None, None

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        h, w, _ = color_image.shape
        width, height = 1000, int(1000 * (h / w))
        img_resized = cv2.resize(color_image, (width, height), interpolation=cv2.INTER_CUBIC)

        # Detect ArUco markers
        corners, ids, _ = cv2.aruco.detectMarkers(img_resized, self.arucoDict, parameters=self.arucoParams)
        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_size, self.cameraMatrix, self.distCoeffs)
            detected_markers, ur_degree, distance = self.aruco_display(corners, ids, img_resized, rvecs, tvecs, w, h, width, height, depth_frame)
        else:
            detected_markers, ur_degree, distance = self.aruco_display(corners, ids, img_resized, None, None, w, h, width, height, depth_frame)

        return detected_markers, ur_degree, distance

    def release(self):
        # Stop streaming and release resources
        self.pipeline.stop()
        cv2.destroyAllWindows()



