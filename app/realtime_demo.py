from ultralytics import YOLO
import cv2

# 모델 불러오기
model = YOLO("models/best.pt")

# 웹캠 실행
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # 화재 감지
    results = model(frame)

    # 결과 시각화
    annotated_frame = results[0].plot()

    # 화면 출력
    cv2.imshow("Fire Detection", annotated_frame)

    # q 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()