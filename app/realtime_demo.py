import cv2
from ultralytics import YOLO
from datetime import datetime
import os

# YOLO 모델 불러오기
model = YOLO("models/best.pt")

# 결과 저장 폴더 생성
os.makedirs("results/screenshots", exist_ok=True)

# 웹캠 열기
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

print("실시간 화재 감지 시작")
print("ESC 키를 누르면 종료됩니다.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # 화재 감지
    results = model.predict(
        source=frame,
        conf=0.25,
        verbose=False
    )

    # 결과 시각화
    annotated_frame = results[0].plot()

    fire_detected = False

    # 탐지 결과 확인
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]

        if class_name in ["fire", "smoke"]:
            fire_detected = True

    # 경고 표시
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
            "WARNING: FIRE DETECTED",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        # 스크린샷 저장
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"results/screenshots/fire_{now}.jpg"
        cv2.imwrite(save_path, annotated_frame)

    # 화면 출력
    cv2.imshow("Real-Time Fire Detection", annotated_frame)

    # ESC 종료
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()