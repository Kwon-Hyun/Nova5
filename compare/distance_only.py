import cv2 as cv
import numpy as np
from ultralytics import YOLO

# YOLOv8 모델 로드
model = YOLO('model/best.pt')

# QR코드 실제 크기 (단위: m, 가정값으로 0.06m 사용)
QR_ACTUAL_SIZE = 0.06

# 두 점 사이 거리 계산 함수
def cv_distance(P, Q):
    return np.sqrt((P[0] - Q[0]) ** 2 + (P[1] - Q[1]) ** 2)

# QR 코드 탐지 및 중심 좌표 계산 함수
def detect_qr_with_yolo(image, boxes, camera_matrix, dist_coeffs):
    for box in boxes:
        xyxy = box.xyxy.cpu().detach().numpy().tolist()[0]
        x1, y1, x2, y2 = map(int, xyxy)

        # YOLO b-box 가로, 세로 크기
        b_width = abs(x2 - x1)
        b_height = abs(y2 - y1)
        qr_center = ((x1 + x2) // 2, (y1 + y2) // 2)

        # Camera 화면 center 좌표 계산
        frame_center = (image.shape[1] // 2, image.shape[0] // 2)

        # Camera 중심 표시
        cv.circle(image, frame_center, 5, (255, 255, 255), -1)
        cv.putText(image, "Camera Center", (frame_center[0] + 10, frame_center[1]),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # QR 코드 중심 표시
        cv.circle(image, qr_center, 5, (0, 255, 255), -1)
        cv.putText(image, f"QR Center: {qr_center}", (qr_center[0] + 10, qr_center[1]),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # 3D 거리 추정: b-box 크기와 QR 코드 실제 크기 사용
        focal_length = camera_matrix[0, 0]  # Focal length from camera matrix
        qr_pixel_size = (b_width + b_height) / 2  # QR 코드 평균 크기 (pixel)
        if qr_pixel_size > 0:
            distance_z = (QR_ACTUAL_SIZE * focal_length) / qr_pixel_size
        else:
            distance_z = None

        # 결과 표시
        cv.putText(image, f"Distance Z: {distance_z:.2f}m", (qr_center[0], qr_center[1] + 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return image

# 실시간 카메라 QR 코드 탐지
def camera_qr_detection():
    cap = cv.VideoCapture(0)  # 내장 카메라 사용
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    # 가정된 카메라 매트릭스와 왜곡 계수 (실제 카메라 캘리브레이션 필요)
    camera_matrix = np.array([[1000, 0, 640],
                              [0, 1000, 360],
                              [0, 0, 1]], dtype=float)
    dist_coeffs = np.zeros((4, 1))  # 왜곡 계수 초기화

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        # YOLOv8 모델로 객체 탐지
        results = model.predict(frame)
        boxes = results[0].boxes

        # 바운딩 박스를 기반으로 QR 코드 중심 좌표 및 거리 계산
        annotated_frame = detect_qr_with_yolo(frame, boxes, camera_matrix, dist_coeffs)

        # 결과 출력
        cv.imshow('YOLOv8 QR Detection', annotated_frame)

        # 'q' 키를 누르면 종료
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    camera_qr_detection()