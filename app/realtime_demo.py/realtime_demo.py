import cv2
from ultralytics import YOLO
from datetime import datetime
import os

# Load trained YOLO model
# 학습된 YOLO 모델 불러오기
model = YOLO("models/best.pt")

# Create results folder
# 결과 저장 폴더 생성
os.makedirs("results/screenshots", exist_ok=True)

# Open webcam
# 웹캠 열기
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라를 열 수 없습니다. / Cannot open camera.")
    exit()

print("Real-time fire detection started.")
print("Press ESC to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("프레임을 읽을 수 없습니다. / Cannot read frame.")
        break

    # Run prediction
    # YOLO 예측 실행
    results = model.predict(
        source=frame,
        conf=0.5,
        verbose=False
    )

    # Draw detection results
    # 탐지 결과 박스 그리기
    annotated_frame = results[0].plot()

    fire_detected = False

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]

        # fire or smoke detection
        # 화재 또는 연기 감지
        if class_name in ["fire", "smoke"]:
            fire_detected = True

    # Warning UI
    # 경고 화면 표시
    if fire_detected:
        cv2.rectangle(
            annotated_frame,
            (0, 0),
            (annotated_frame.shape[1], 80),
            (0, 0, 255),
            -1
        )

        cv2.putText(
            annotated_frame,
            "WARNING: FIRE RISK DETECTED",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        # Save screenshot
        # 위험 감지 화면 저장
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"results/screenshots/fire_detected_{now}.jpg"
        cv2.imwrite(save_path, annotated_frame)

    cv2.imshow("Real-Time Fire Detection", annotated_frame)

    # ESC key to quit
    # ESC 키를 누르면 종료
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
