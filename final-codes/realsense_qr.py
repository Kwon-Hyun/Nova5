import pyrealsense2 as rs
import numpy as np
import cv2 as cv
import math
import os
import time
import timeit

# QR 코드 방향 정의
CV_QR_UP = "위쪽"   # 북 0
CV_QR_RIGHT = "오른쪽"  # 동 1
CV_QR_DOWN = "아래쪽"  # 남 2
CV_QR_LEFT = "왼쪽"  # 서 3

# 두 점 사이 거리 계산 함수
def cv_distance(P, Q):
    return np.sqrt((P[0] - Q[0]) ** 2 + (P[1] - Q[1]) ** 2)

# 선분 LM을 기준으로 점 J에서 수직으로 떨어진 거리 계산 함수
def cv_lineEquation(L, M, J):
    a = -(M[1] - L[1]) / (M[0] - L[0])
    b = 1.0
    c = ((M[1] - L[1]) / (M[0] - L[0])) * L[0] - L[1]
    pdist = (a * J[0] + (b * J[1]) + c) / np.sqrt(a * a + b * b)
    return pdist

# 두 점으로 이루어진 선의 기울기 계산 함수
def cv_lineSlope(L, M):
    dx = M[0] - L[0]
    dy = M[1] - L[1]
    if dy != 0:
        alignment = 1
        return dy / dx, alignment
    else:
        alignment = 0
        return 0.0, alignment

# QR 코드 내 3개의 Position Pattern을 이용하여 방향 결정하는 함수
def find_qr_orientation(contours, mc):
    AB = cv_distance(mc[0], mc[1])
    BC = cv_distance(mc[1], mc[2])
    CA = cv_distance(mc[2], mc[0])

    if AB > BC and AB > CA:
        outlier = 2
        median1 = 0
        median2 = 1
    elif CA > AB and CA > BC:
        outlier = 1
        median1 = 0
        median2 = 2
    else:
        outlier = 0
        median1 = 1
        median2 = 2

    dist = cv_lineEquation(mc[median1], mc[median2], mc[outlier])
    slope, align = cv_lineSlope(mc[median1], mc[median2])

    if align == 0:
        orientation = CV_QR_UP
    elif slope < 0 and dist < 0:
        orientation = CV_QR_UP
    elif slope > 0 and dist < 0:
        orientation = CV_QR_RIGHT
    elif slope < 0 and dist > 0:
        orientation = CV_QR_DOWN
    elif slope > 0 and dist > 0:
        orientation = CV_QR_LEFT

    return outlier, median1, median2, orientation, slope

# QR 코드 tilt 기울기값 계산 위한 함수
def calculate_tilt(points):
    if points is None:
        return None

    # tilt 계산법1 - Homography 연산을 사용하는 방법
    dst_pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype="float32")  # QRcode가 정사각형이라는 전제로.
    h, _ = cv.findHomography(points[0], dst_pts)

    # Homography 행렬로부터 기울기 값 추출
    if h is not None:
        tilt_angle = np.degrees(np.arctan2(h[2, 1], h[2, 2]))
        
        return tilt_angle
    return None


# QR 코드 위치 패턴 및 방향 감지 함수
def detect_qr(image):
    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, img_bin = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    img_morph = cv.morphologyEx(img_bin, cv.MORPH_OPEN, kernel)
    img_canny = cv.Canny(img_gray, 100, 200)
    contours, hierarchy = cv.findContours(img_canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


    qr_detector = cv.QRCodeDetector()
    data, points, _ = qr_detector.detectAndDecode(image)

    mark = 0
    A, B, C = None, None, None

    for i in range(len(contours)):
        k = i
        c = 0
        while hierarchy[0][k][2] != -1:
            k = hierarchy[0][k][2]
            c += 1
        if hierarchy[0][k][2] != -1:
            c += 1
        if c >= 5:
            if mark == 0:
                A = i
            elif mark == 1:
                B = i
            elif mark == 2:
                C = i
            mark += 1

    # rotation하는 거는 distance하면서 좀 더 수정봐야할 거 같음.
    if mark >= 3:
        mu = [cv.moments(contours[A]), cv.moments(contours[B]), cv.moments(contours[C])]
        mc = [(mu[i]["m10"] / mu[i]["m00"], mu[i]["m01"] / mu[i]["m00"]) for i in range(3)]
        outlier, bottom, right, orientation, slope = find_qr_orientation(contours, mc)

        print(f"QR 코드 방향: {orientation}")
        rotation_angle = np.degrees(np.arctan(slope)) if slope >= 0 else 360 + np.degrees(np.arctan(slope))
        print(f"QR code 기울기(회전각) : {rotation_angle:.2f} 도")

        if orientation == CV_QR_UP:
            print(f"{rotation_angle:.2f}만큼 돌리세요.")
        elif orientation == CV_QR_DOWN:
            print(f"시계 방향으로 {rotation_angle:.2f}만큼 돌리세요.")
        elif orientation == CV_QR_RIGHT:
            print(f"{CV_QR_LEFT} 방향으로 {rotation_angle:.2f}만큼 돌리세요.")
        else:
            print(f"{CV_QR_RIGHT} 방향으로 {rotation_angle:.2f}만큼 돌리세요.")
        

        # QR tilt 기울기 게산
        tilt_angle = calculate_tilt(points)

        if tilt_angle is not None:
            cv.putText(image, f"Tilt값 : {tilt_angle:.2f}도", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            print(f"QR code Tilt값 : {tilt_angle:.2f}도")


        # 빗변의 중심점 계산 (빗변의 중심점이 qr의 center일거라는 가정 때문.)
        qr_center = (
            int((mc[bottom][0] + mc[right][0]) / 2),
            int((mc[bottom][1] + mc[right][1]) / 2)
        )

        # QR 코드 중심에 점 찍기 및 좌표 표시
        cv.circle(image, qr_center, 5, (0, 255, 255), -1)
        cv.putText(image, f"Center: {qr_center}", (qr_center[0] + 10, qr_center[1]),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        print(f"QR code center : {qr_center}")

        # Camera 화면 center 좌표 계산
        frame_center = (image.shape[1] // 2, image.shape[0] // 2)
        cv.circle(image, frame_center, 5, (255, 255, 255), -1)
        cv.putText(image, "Camera Center", (frame_center[0] + 10, frame_center[1]),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # QR 코드 중심과 카메라 중심 거리 계산
        distance_to_center = cv_distance(frame_center, qr_center)
        print(f"QRcode center <-> Camera center Distance : {distance_to_center:.2f} pixel")

        # QR 코드가 중앙에 가까운지 판단
        threshold_center_distance = 20  # 임계값 설정 (30 pixel 내외) - threshold 이하면 중앙값에 있다고 판단.
        if distance_to_center < threshold_center_distance:
            cv.putText(image, "QR <-> camera center Ok!!", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv.putText(image, "Not camera center..", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


        # QR size (추후 distance 판단 위함)
        # 외곽 사각형 그리기 & 가로, 세로 pixel 길이 측정
        # minAreaRect, polyline, ... 등 방법 여러 가지 비교해보기
        # minAreaRect - https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html 참고
        '''
        cnt = contours[outlier]
        rect = cv.minAreaRect(cnt)
        #rect = cv.minAreaRect(np.array([mc[bottom], mc[right], mc[outlier]]))
        box = cv.boxPoints(rect)
        box = np.int0(box)
        
        cv.drawContours(image, [box], 0, (0,0,255), 2)

        width = int(rect[1][0])
        height = int(rect[1][1])
        print(f"QR 외곽 사각형 가로 : {width}pixel, 세로 : {height}pixel")

        # 외곽 사각형 가로세로 길이 표시
        cv.putText(image, f"가로 : {width} pixel", (box[0][0], box[0][1] - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv.putText(image, f"세로 : {height} pixel", (box[0][0], box[0][1] - 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        '''

        # Camera와 QRcode의 Distance
        if data:
            print(f"QR Code Data: {data}")
            points = points[0].astype(int)
            
            # QR b-box 그리기
            for i in range(4):
                cv.line(image, tuple(points[i]), tuple(points[(i + 1) % 4]), (255, 0, 0), 3)
            
            # QR b-box의 가로, 세로 길이 계산
            width = int(cv_distance(points[0], points[1]))
            height = int(cv_distance(points[1], points[2]))
            print(f"QR b-box 가로 : {width} pixel, 세로 : {height} pixel")
            
            # QR b-box 크기 표시
            cv.putText(image, f"Width : {width} pixel", (points[0][0], points[0][1] - 10),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv.putText(image, f"Height : {height} pixel", (points[0][0], points[0][1] - 30),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # QR 코드 4cm x 4cm 크기를 기준값으로 설정하여 distance 측정
            target_size_pixel = 151  # 4cm = 약 151pixel
            
            width_size_difference = width - target_size_pixel
            height_size_difference = height - target_size_pixel
            size_difference = (width + height) / 2 - target_size_pixel  # 일단 지금은 1차적으로 가로세로 평균크기로 비교
            
            if width_size_difference > 0 or height_size_difference > 0:
                print(f"QR 코드가 {abs(size_difference):.2f} pixel 거리만큼 뒤로 ㄱㄱ (멀어지기)")
            elif width_size_difference > 0 or height_size_difference < 0:
                print(f"rotation에 신경써서 수직이 되도록.")
            else:
                print("굳굳굳")


            #elif width_size_difference < 0 or  
            '''
            if size_difference > 0:
                print(f"QR 코드가 {abs(size_difference):.2f} pixel 거리만큼 뒤로 ㄱㄱ (멀어지기)")
            elif size_difference < 0:
                print(f"QR 코드가 {abs(size_difference):.2f} pixel 거리만큼 앞으로 ㄱㄱ (다가가기)")
            
            else:
                print("지금 QR이 4cm로 보이므로, 거리 괜찮다고 판단 !! :-)")
            '''
            print("\n")

        # position pattern - 외곽선 그리기
        cv.drawContours(image, contours, A, (0, 255, 0), 2) # green
        cv.drawContours(image, contours, B, (255, 0, 0), 2) # red
        cv.drawContours(image, contours, C, (0, 0, 255), 2) # blue

        return image
    return None

# 이미지 저장 함수
def save_image(image, folder="qr_detection_results"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = f"241119_qr_detection_{int(time.time())}.jpg"
    filepath = os.path.join(folder, filename)
    cv.imwrite(filepath, image)

# RealSense 카메라로 QR 코드 감지
def realsensecam_qr_detection():
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 6)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 6)
    pipe.start(cfg)

    while True:
        frames = pipe.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        processed_frame = detect_qr(color_image)

        if processed_frame is not None:
            images = np.hstack((processed_frame, cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.03), cv.COLORMAP_JET)))
        else:
            images = np.hstack((color_image, cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.03), cv.COLORMAP_JET)))

        cv.imshow('QR Code Detection (RGB + Depth)', images)

        save_image(images)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    pipe.stop()
    cv.destroyAllWindows()

if __name__ == "__main__":
    realsensecam_qr_detection()