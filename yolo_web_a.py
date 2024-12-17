import torch
import threading
import cv2
import numpy as np
import time
from pathlib import Path
import pathlib
import os
import requests

pathlib.PosixPath = pathlib.WindowsPath
import warnings
warnings.filterwarnings('ignore')


# Load the Model
path = Path("best.pt")
model = torch.hub.load('ultralytics/yolov5', 'custom', path, 
                    #    pretrained=True
                       )
min_confidence = 0.5  # 최소 confidence 설정

# 영상 경로 (CCTV 스트q림 또는 파일 경로)
video_path = "http://210.99.70.120:1935/live/cctv001.stream/playlist.m3u8"  # CCTV 스트림

# 비디오 캡처
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print('--(!)Error opening video capture')
    exit(0)

# 이벤트 객체 생성 (값이 변경되면 신호를 줌)
value_changed_event = threading.Event()

# 전역 변수로 이전 값과 현재 값을 저장
previous_car_count = None
previous_parking_lot = None

current_car_count = 0
current_parking_lot = 20

# YOLOv5의 출력 처리 및 박스 그리기 함수
def detectAndDisplay(frame):
    global current_car_count, current_parking_lot, previous_car_count, previous_parking_lot
    height, width, channels = frame.shape

    # YOLOv5에 이미지를 전달하고 결과 받아오기 (추론)
    results = model(frame)

    # 탐지된 결과들 가져오기
    labels, confidences, boxes = [], [], []
    car_count = 0
    parking_lot = 8

    for *box, conf, class_id in results.xyxy[0].cpu().numpy():
        x, y, w, h = map(int, box)
        confidence = float(conf)

        # Confidence threshold 적용
        if confidence > min_confidence:
            labels.append(model.names[int(class_id)])
            confidences.append(confidence)
            boxes.append([x, y, w - x, h - y])  # YOLOv5의 좌표는 좌상단(x,y), 우하단(w,h)

            # 주차장의 빈자리와 자동차 댓수 파악
            if model.names[int(class_id)] == 'car':
                car_count += 1
            elif model.names[int(class_id)] == 'truck':
                car_count += 1
            elif model.names[int(class_id)] == 'bus':
                car_count += 1
            elif model.names[int(class_id)] == 'space-empty':
                parking_lot -= 1

    # 현재 값 업데이트
    current_car_count = car_count
    current_parking_lot = parking_lot

    # 값이 변경되었을 때만 이벤트 발생
    if current_car_count != previous_car_count or current_parking_lot != previous_parking_lot:
        value_changed_event.set()  # 이벤트 신호를 보냄

    # 탐지된 객체를 프레임에 그리기
    for i, (label, confidence, box) in enumerate(zip(labels, confidences, boxes)):
        x, y, w, h = box
        color = (0, 255, 0)  # 박스 컬러 (초록색)
        label_text = f"{label}: {confidence:.2f}"

        # 박스 그리기
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 결과 이미지 출력
    cv2.imshow("YOLOv5 Detection", frame)
    
    # time.sleep(2)

# 값이 변경되었을 때 처리하는 함수
def handle_value_change():
    global previous_car_count, previous_parking_lot, current_car_count, current_parking_lot
    while True:
        # 이벤트가 발생할 때까지 대기
        value_changed_event.wait()

        # Flask 서버로 데이터를 전송
        data = {
            "car_count": current_car_count,
            "empty_spots": 8 - current_car_count
        }
        response = requests.post("http://localhost:5000/update_parking_info", json=data)
        print(f"서버 응답: {response.text}")

        # 이전 값을 현재 값으로 갱신
        previous_car_count = current_car_count
        previous_parking_lot = current_parking_lot

        # 이벤트를 다시 초기화
        value_changed_event.clear()

# 영상 처리 루프
def video_loop():
    while True:
        ret, frame = cap.read()
        if frame is None:
            print('--(!) No captured frame -- Break!')
            break

        # YOLOv5 탐지 및 박스 그리기
        detectAndDisplay(frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 두 스레드를 실행: 하나는 영상을 처리하고, 하나는 값을 출력하는 역할
monitor_thread = threading.Thread(target=video_loop)
handler_thread = threading.Thread(target=handle_value_change)

monitor_thread.start()
handler_thread.start()

monitor_thread.join()
handler_thread.join()

# 리소스 해제
cap.release()
cv2.destroyAllWindows()