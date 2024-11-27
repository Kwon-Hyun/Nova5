import cv2 as cv
import numpy as np
from ultralytics import YOLO
import math

# YOLOv8 모델 로드
model = YOLO('model/best.pt')

# 두 점 사이 거리 계산 함수
def cv_distance(P, Q):
    return np.sqrt((P[0] - Q[0]) ** 2 + (P[1] - Q[1]) ** 2)

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

# QR 코드 탐지 및 중심 좌표 계산 함수
def detect_qr_with_yolo(image, boxes):
    for box in boxes:
        xyxy = box.xyxy.cpu().detach().numpy().tolist()[0]
        confidence = box.conf.cpu().detach().numpy().tolist()
        class_id_list = box.cls.cpu().detach().numpy().tolist()

        # class_id_list가 비어있지 않다면 첫 번째 요소를 사용
        if class_id_list:
            class_id = int(class_id_list[0])
        else:
            class_id = None  # 또는 적절한 기본값 설정

        # b-box 좌표 추출
        x1, y1, x2, y2 = map(int, xyxy)

        # yolo b-box이자 QR의 중심 좌표 계산
        qr_center = ((x1 + x2) // 2, (y1 + y2) // 2)

        # Camera 화면 center 좌표 계산
        frame_center = (image.shape[1] // 2, image.shape[0] // 2)
        
        # center 표시
        cv.circle(image, frame_center, 5, (255, 255, 255), -1)
        cv.putText(image, "Camera Center", (frame_center[0] + 10, frame_center[1]),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # QR 중심과 camera 중심 거리 - 순수 distance pixel 하나 구할 때 (이건 아마 안 쓸 듯)
        #distance_to_center = cv_distance(frame_center, qr_center)
        #print(f"QRcode center <-> Camera center Distance : {distance_to_center:.2f} pixel")
        
        # QR 코드 중심과 카메라 중심 거리 계산 - (x, y) 형식
        center_distance_x = qr_center[0] - frame_center[0]
        center_distance_y = qr_center[1] - frame_center[1]
        print(f"QRcode center <-> Camera center Distance : {center_distance_x}, {center_distance_y} pixel")

        # 회전 각도 계산 (대각선 기준)
        slope, _ = cv_lineSlope((x1, y1), (x2, y2))
        rotation_angle = np.degrees(np.arctan(slope))

        # 결과 출력
        cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.circle(image, qr_center, 5, (0, 255, 255), -1)
        cv.putText(image, f"Center : {qr_center}", (qr_center[0] + 10, qr_center[1]),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv.putText(image, f"Rotation (not tilt) : {rotation_angle:.2f}", (10, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv.putText(image, f"Distance to center (x, y) : ({center_distance_x}, {center_distance_y})", (10, 80), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return image

# 실시간 카메라 QR 코드 탐지
def camera_qr_detection():
    cap = cv.VideoCapture(0)  # 내장 카메라 사용
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break
        
        # YOLOv8 모델로 객체 탐지
        results = model.predict(frame)
        # tracking으로 객체 탐지
        #tracking_results = model.track(source=0, show=True, tracker="default.yaml")
        annotated_frame = results[0].plot()
        # b-box 정보 알고 싶으면, 아래와 같이.
        boxes = results[0].boxes

        # 바운딩 박스를 기반으로 QR 코드 중심 좌표 계산
        annotated_frame = detect_qr_with_yolo(annotated_frame, boxes)

        # 결과 출력
        cv.imshow('YOLOv8 QR Detection', annotated_frame)

        # 'q' 키를 누르면 종료
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    camera_qr_detection()