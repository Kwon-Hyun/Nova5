# Nova5 & OHT & AMR
- Dobot Nova5 (with Realsense Depth Camera, RaspberryPi Camera Module v3 etc.) - QR detection &amp; Decoding
- AMR with SLAM (Simultaneous Localization and Mapping) - QR detection &amp; Decoding
- OHT (forking, picking system with 3 camera) - QR detection &amp; Decoding


## Category 분류
<b> 1. TCP/IP socket 통신</b>
<br>
    - create.py <br>
    - detect.py <br>
    - dobot.py <br>
    - dobotQR.py <br>

<b> 2. YOLOv8 model training</b>
<br>
    - best.pt (training by 1,027 QR images) <br>

<b> 3. test & compare</b>
<br>
    - mac_test.py : mac 내장 webcam test <br>
    - yolo_test.py : yolo model & 기존 Algorithm 적용 test <br>

<b> * 최종 Source Code File</b>
<br>
    - realsense_qr.py (최종본!!)

## 진행 상황
1. Algorithm
    - QRcode Position Pattern contouring
    - Nova5와 .py 연결 완료 (dobotQR.py)
    - QRcode Rotation, QR center detection
    - QRcode center detection & Camera center와 QR center 일치시키기 완료
    - QRcode와 Camera 간 Distance 구하기 (기준 QR size : 6cm*6cm (170pixel))

2. with YOLOv8
    - QRcode Detection & Decoding (-> 아주 good ㅋㅋ)
    - QRcode center positioning

    <해야할 것>
    - QRcode distance
    - QR (x, y, z)
    - <b> SLAM path planning i.n.g . . . . . . .</b>