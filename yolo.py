import cv2
import torch
from ultralytics import YOLO
from flask import Flask, render_template, jsonify, Response

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

app = Flask(__name__)

# YOLO 모델 불러오기 (커스텀 가중치 사용)
model = YOLO('yolov5-master/yolov5-master/best_1017.pt')

# 'http://192.168.0.3:5000/video_feed'

# CCTV 스트림 경로 또는 mp4 파일 경로 설정
video_path = "http://192.168.0.109:5000/video_feed"  # CCTV 스트림
cap = cv2.VideoCapture(video_path)

# 전역 변수로 탐지 정보 저장
detection_info = "탐지된 객체가 없습니다."

# 프레임을 읽고 객체 탐지를 수행한 결과를 반환하는 함수
def generate_frames():
    global detection_info

    while True:
        ret, frame = cap.read()
        if not ret:
            print("캠을 열 수 없습니다!")
            break

        # YOLO 모델을 통해 객체 탐지 수행
        results = model.predict(frame)

        # 첫 번째 인식된 객체의 클래스와 확률을 가져옴
        if results[0].boxes:  # 객체가 탐지된 경우
            obj = results[0].boxes[0]  # 첫 번째 객체
            class_id = int(obj.cls[0])  # 클래스 ID
            confidence = obj.conf[0]  # 신뢰도 (확률)

            # YOLO 모델에서 클래스 이름을 가져옴
            class_name = model.names[class_id]  # 클래스 이름 가져오기
            detection_info = f"{class_name} ({confidence * 100:.2f}%)"

            # 결과 이미지 업데이트
            result_img = results[0].plot()
        else:
            result_img = frame
            detection_info = "탐지된 객체가 없습니다."

        # OpenCV 이미지 포맷을 JPG로 변환
        ret, buffer = cv2.imencode('.jpg', result_img)
        frame = buffer.tobytes()

        # Flask를 사용하여 프레임을 반환
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# 실시간 탐지 정보를 반환하는 라우트 추가
@app.route('/get_detection_info')
def get_detection_info():
    global detection_info
    return jsonify({'detection_info': detection_info})

@app.route('/')
def index():
    return render_template('flask_face.html')

# /video_feed 라우트에서 비디오 스트리밍을 제공
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)

