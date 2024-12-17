from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# 주차 정보 저장용 전역 변수
parking_info = {
    "car_count": 0,
    "empty_spots": 8
}

# 메인 페이지 라우트
@app.route('/')
def index():
    return render_template('yolo_web.html', parking_info=parking_info)

# A 파일에서 데이터를 받는 라우트
@app.route('/update_parking_info', methods=['POST'])
def update_parking_info():
    global parking_info
    data = request.get_json()
    parking_info["car_count"] = data["car_count"]
    parking_info["empty_spots"] = data["empty_spots"]
    return jsonify({"message": "주차 정보가 업데이트되었습니다."})

if __name__ == "__main__":
    app.run(debug=True)