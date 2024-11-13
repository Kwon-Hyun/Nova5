# Nova5
Dobot Nova5 (with Realsense Depth Camera, AMB82 mini board etc.) - QR detection &amp; Decoding

## Category 분류
1. TCP/IP socket 통신
    - create.py
    - detect.py
    - dobot.py
    - dobotQR.py
    - realsense_qr.py (최종본)

## 진행 상황
- QR code Position Pattern contouring
- Nova5와 .py 연결 완료 (dobotQR.py)
- QRcode Rotation, QR center detection (realsense_qr.py)
- QRcode center detection & Camera center와 QR center 일치시키기 완료
- QRcode와 Camera 간 Distance 구하기 (기준 QR size : 4cm*4cm (151pixel))
