# Nova5 & OHT & AMR
- Dobot Nova5 (with Realsense Depth Camera, RaspberryPi Camera Module v3 etc.) - QR detection &amp; Decoding
- AMR with SLAM (Simultaneous Localization and Mapping) - QR detection &amp; Decoding
- OHT (forking, picking system with 3 camera) - QR detection &amp; Decoding

---

## Category 분류


project/


├── final-codes


		├── realsense_qr.py ( for macbook webcam test )

		├── yolo_realsense.py ( for 최종 코드들 합치기용 )

  
├── for-preview


		├── pattern_review.py ( QR pattern 기반 detection ) 

		├── yolov8_review.py ( YOLOv8 custom model 기반 detection & 3D distance )

  
├── img


		├── l2m_qr.png ( qr_create.py 통해서 생성한 img )

  
├── model


		├── L2M_QR_YOLO.ipynb ( for training yolo model )
  
		├── best.pt ( training 완료된 QR YOLOv8 custom model )
  
		├── data.yaml ( train model에 대한 data file )

  
├── socket ( for TCP/IP socket 통신 with Nova5 )


		├── dobot_socket.py


├── create


		├── qr_create_v1.py
  
		├── qr_create_v2.py


└── .gitignore

└── README.md

└── nova5_yolo.py ( 실행시킬 final code )


---

## Installation
**- ultralytics**

  [참고](https://dagshub.com/Ultralytics/ultralytics/src/2b49d71772ae8e2a5ccede2127430816503bf469/docs/ko/quickstart.md)

  ```
  pip install ultralytics
  ```
  
**- pyrealsense2**

  [참고1](https://support.intelrealsense.com/hc/en-us/community/posts/26334072305171-pip-install-pyrealsense2-not-working)
  
  [참고2](https://velog.io/@zzziito/Realsense-%ED%8C%8C%EC%9D%B4%EC%8D%AC%EC%9C%BC%EB%A1%9C-%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0)

  ```
  # (Window) Python 3.7 ~ Python 3.11 ver 까지
  pip install pyrealsense2
  ```

  
**- numpy**

  [참고](https://carpfish.tistory.com/entry/pip%EB%A5%BC-%EC%9D%B4%EC%9A%A9%ED%95%9C-Numpy-%EC%84%A4%EC%B9%98python3)

  ```
  # Python version 확인
  python3 -V
  ```
  ```
  # pip version 확인
  pip3 -V
  ```
  ```
  # Python3.XX인 경우
  pip3 install numpy
  ```

  
**- opencv**

  [참고](https://foss4g.tistory.com/1500)

  ```
  # 주요 module install
  pip install opencv-python
  ```
  ```
  # 주요 module 및 추가 module install
  pip install opencv-contrib-python
  ```

  
**- pyzbar**

  [참고](https://pypi.org/project/pyzbar/)

  ```
  pip install pyzbar
  ```

  
**- qrcode**

  [참고](https://pypi.org/project/qrcode/)

  ```
  pip install qrcode
  ```


---

1. TCP/IP socket 통신
    - create.py
    - detect.py
    - dobot.py
    - dobotQR.py

2. YOLOv8 model training
    - best.pt (training by 1,027 QR images)

3. test & compare
    - mac_test.py : mac 내장 webcam test
    - yolo_test.py : yolo model & 기존 Algorithm 적용 test

**최종 Source Code File**

    nova5_yolo.py (최종본!!)

---

## QR 탐지 알고리즘
**1. QR Pattern Algorithm (`pattern_review.py`)**
- QRcode Position Pattern contouring
- Nova5와 .py 연결 완료 (dobotQR.py)
- QRcode Rotation, QR center detection
- QRcode center (x, y) 추출 가능
- QRcode center detection & Camera center와 QR center 일치시키기 완료
- QRcode와 Camera 간 Distance 구하기 완료 (기준 QR size : 6cm*6cm (170pixel))
- QRcode 실시간 Distance(QRcode z값) 계산 가능 (단위 : m)

**2. YOLOv8 custom model Algorithm (`yolov8_review.py`)**
- QRcode Detection & Decoding (-> 아주 good)
- QRcode center positioning
- ver1 : 내장 라이브러리 사용, ver2 : QR Position Pattern, ver3 : YOLOv8 custom training 에 대한 성능 비교 분석
