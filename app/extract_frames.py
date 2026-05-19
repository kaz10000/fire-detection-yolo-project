import cv2
import os

# 입력 영상 경로
video_path = "test_assets/drone_test.mp4"

# 프레임 저장 폴더
output_dir = "data/raw/drone_smoke_frames"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("영상 파일을 열 수 없습니다.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps / 2)  # 1초에 약 2장 추출

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if frame_count % frame_interval == 0:
        filename = f"drone_smoke_{saved_count:04d}.jpg"
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, frame)
        saved_count += 1

    frame_count += 1

cap.release()

print(f"프레임 추출 완료: {saved_count}장 저장")
print(f"저장 위치: {output_dir}")
