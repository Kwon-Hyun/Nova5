from ultralytics import YOLO
import cv2

# YOLOv8 모델 로드
model = YOLO('model/best.pt')

# 카메라 설정
cap = cv2.VideoCapture(0)  # 0은 기본 카메라를 의미함

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임에 대해 객체 탐지 수행
    results = model.predict(frame)

    # 결과 시각화
    annotated_frame = results[0].plot()

    # 프레임 출력
    cv2.imshow('YOLOv8 Detection', annotated_frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) == ord('q'):
        break

# 카메라 끄기
cap.release()
cv2.destroyAllWindows()