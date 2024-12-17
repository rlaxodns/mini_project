import cv2
from flask import Flask, Response

app = Flask(__name__)

# 웹캠 초기화
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()  # 카메라에서 프레임 읽기
        if not success:
            break
        else:
            # 프레임을 JPEG 형식으로 인코딩
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue  # 인코딩 실패 시 다음 프레임으로 넘어감
            
            frame = buffer.tobytes()  # 바이트 배열로 변환

            # MJPEG 스트리밍 형식으로 반환
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')  # "\r\n\r\n"를 추가하여 스트리밍을 정상화

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)  # threaded=True 추가
