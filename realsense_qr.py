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

# QR 코드 위치 패턴 및 방향 감지 함수
def detect_qr(image):
    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, img_bin = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    img_morph = cv.morphologyEx(img_bin, cv.MORPH_OPEN, kernel)
    img_canny = cv.Canny(img_gray, 100, 200)
    contours, hierarchy = cv.findContours(img_canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

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

    if mark >= 3:
        mu = [cv.moments(contours[A]), cv.moments(contours[B]), cv.moments(contours[C])]
        mc = [(mu[i]["m10"] / mu[i]["m00"], mu[i]["m01"] / mu[i]["m00"]) for i in range(3)]
        outlier, bottom, right, orientation, slope = find_qr_orientation(contours, mc)

        print(f"QR 코드 방향: {orientation}")
        rotation_angle = np.degrees(np.arctan(slope)) if slope >= 0 else 360 + np.degrees(np.arctan(slope))
        print(f"QR code 기울기(회전각) : {rotation_angle:.2f} 도")

        cv.drawContours(image, contours, A, (0, 255, 0), 2)
        cv.drawContours(image, contours, B, (255, 0, 0), 2)
        cv.drawContours(image, contours, C, (0, 0, 255), 2)
        return image
    return None

# RealSense 카메라로 QR 코드 감지
def realtime_qr_detection():
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

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    pipe.stop()
    cv.destroyAllWindows()

if __name__ == "__main__":
    realtime_qr_detection()