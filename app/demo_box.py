import cv2
import os
import time
from pathlib import Path
from datetime import datetime
from tkinter import Tk, filedialog

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


# =========================
# 기본 설정
# =========================

MODEL_PATH = "models/best.pt"
OUTPUT_DIR = "results/demo_outputs"

WINDOW_NAME = "AI 실시간 화재 감지 안전 시스템"

CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 720

VIDEO_AREA_X = 40
VIDEO_AREA_Y = 120
VIDEO_AREA_W = 820
VIDEO_AREA_H = 520

PANEL_X = 900
PANEL_Y = 120
PANEL_W = 340
PANEL_H = 520

CONF_THRESHOLD = 0.15
DANGER_CLASSES = ["fire", "smoke", "Fire", "Smoke", "flame", "Flame"]

WARNING_HOLD_SECONDS = 2.0


# =========================
# 한글 폰트 설정
# =========================

def get_korean_font(size=28):
    font_candidates = [
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/malgunbd.ttf",
        "C:/Windows/Fonts/gulim.ttc",
    ]

    for font_path in font_candidates:
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, size)

    return ImageFont.load_default()


FONT_TITLE = get_korean_font(34)
FONT_HEADER = get_korean_font(28)
FONT_PANEL_TITLE = get_korean_font(30)
FONT_NORMAL = get_korean_font(24)
FONT_SMALL = get_korean_font(20)


def draw_korean_text(image, text, position, font, color=(255, 255, 255)):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)


# =========================
# 파일 선택
# =========================

def select_file():
    root = Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="화재 이미지 또는 영상 파일 선택",
        filetypes=[
            ("이미지/영상 파일", "*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mov *.mkv"),
            ("모든 파일", "*.*")
        ]
    )

    root.destroy()
    return file_path


# =========================
# 이미지 비율 유지 리사이즈
# =========================

def resize_with_padding(image, target_w, target_h):
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h))

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    canvas[:] = (25, 25, 25)

    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas


# =========================
# UI 화면 생성
# =========================

def draw_ui(frame, danger_detected, detected_texts, max_conf, fps=0.0):
    canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
    canvas[:] = (18, 24, 38)

    if danger_detected:
        header_color = (0, 0, 180)
        header_text = "경고: 화재 위험 감지"
    else:
        header_color = (0, 120, 0)
        header_text = "상태: 안전"

    cv2.rectangle(canvas, (0, 0), (CANVAS_WIDTH, 90), header_color, -1)

    canvas = draw_korean_text(
        canvas,
        "AI 실시간 화재 감지 안전 시스템",
        (40, 15),
        FONT_TITLE,
        (255, 255, 255)
    )

    canvas = draw_korean_text(
        canvas,
        header_text,
        (40, 55),
        FONT_HEADER,
        (255, 255, 255)
    )

    cv2.rectangle(
        canvas,
        (VIDEO_AREA_X - 4, VIDEO_AREA_Y - 4),
        (VIDEO_AREA_X + VIDEO_AREA_W + 4, VIDEO_AREA_Y + VIDEO_AREA_H + 4),
        (90, 120, 180),
        2
    )

    display_frame = resize_with_padding(frame, VIDEO_AREA_W, VIDEO_AREA_H)

    canvas[
        VIDEO_AREA_Y:VIDEO_AREA_Y + VIDEO_AREA_H,
        VIDEO_AREA_X:VIDEO_AREA_X + VIDEO_AREA_W
    ] = display_frame

    cv2.rectangle(
        canvas,
        (PANEL_X, PANEL_Y),
        (PANEL_X + PANEL_W, PANEL_Y + PANEL_H),
        (30, 36, 52),
        -1
    )

    cv2.rectangle(
        canvas,
        (PANEL_X, PANEL_Y),
        (PANEL_X + PANEL_W, PANEL_Y + PANEL_H),
        (90, 120, 180),
        2
    )

    canvas = draw_korean_text(
        canvas,
        "감지 정보",
        (PANEL_X + 25, PANEL_Y + 25),
        FONT_PANEL_TITLE,
        (255, 255, 255)
    )

    status_text = "위험" if danger_detected else "안전"
    status_color = (255, 80, 80) if danger_detected else (80, 255, 80)

    canvas = draw_korean_text(
        canvas,
        f"상태: {status_text}",
        (PANEL_X + 25, PANEL_Y + 90),
        FONT_NORMAL,
        status_color
    )

    canvas = draw_korean_text(
        canvas,
        f"최대 신뢰도: {max_conf:.2f}",
        (PANEL_X + 25, PANEL_Y + 135),
        FONT_SMALL,
        (230, 230, 230)
    )

    canvas = draw_korean_text(
        canvas,
        f"처리 속도: {fps:.1f} FPS",
        (PANEL_X + 25, PANEL_Y + 175),
        FONT_SMALL,
        (230, 230, 230)
    )

    canvas = draw_korean_text(
        canvas,
        "감지 항목:",
        (PANEL_X + 25, PANEL_Y + 235),
        FONT_SMALL,
        (255, 255, 255)
    )

    y = PANEL_Y + 275

    if detected_texts:
        for text in detected_texts[:6]:
            canvas = draw_korean_text(
                canvas,
                f"- {text}",
                (PANEL_X + 25, y),
                FONT_SMALL,
                (220, 220, 220)
            )
            y += 34
    else:
        canvas = draw_korean_text(
            canvas,
            "- 감지 없음",
            (PANEL_X + 25, y),
            FONT_SMALL,
            (180, 180, 180)
        )

    canvas = draw_korean_text(
        canvas,
        "ESC 키를 누르면 종료됩니다",
        (PANEL_X + 25, PANEL_Y + PANEL_H - 45),
        FONT_SMALL,
        (180, 180, 180)
    )

    return canvas


# =========================
# YOLO 탐지
# =========================

def run_detection_on_frame(model, frame):
    results = model.predict(
        source=frame,
        conf=CONF_THRESHOLD,
        verbose=False
    )

    annotated_frame = results[0].plot()

    danger_detected = False
    detected_texts = []
    max_conf = 0.0

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        conf = float(box.conf[0])

        if class_name.lower() == "fire":
            display_name = "화재"
        elif class_name.lower() == "smoke":
            display_name = "연기"
        else:
            display_name = class_name

        detected_texts.append(f"{display_name}: {conf:.2f}")
        max_conf = max(max_conf, conf)

        if class_name in DANGER_CLASSES:
            danger_detected = True

    return annotated_frame, danger_detected, detected_texts, max_conf


# =========================
# 메인 실행
# =========================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        print("models 폴더 안에 best.pt 파일을 넣어주세요.")
        return

    print("모델을 불러오는 중입니다...")
    model = YOLO(MODEL_PATH)

    print("모델 클래스:", model.names)

    source_path = select_file()

    if not source_path:
        print("선택된 파일이 없습니다.")
        return

    source_path = Path(source_path)
    ext = source_path.suffix.lower()

    print(f"선택된 파일: {source_path}")

    image_exts = [".jpg", ".jpeg", ".png", ".bmp"]
    video_exts = [".mp4", ".avi", ".mov", ".mkv"]

    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    last_warning_time = 0

    if ext in image_exts:
        image = cv2.imread(str(source_path))

        if image is None:
            print("이미지 파일을 읽을 수 없습니다.")
            return

        annotated_frame, danger_detected, detected_texts, max_conf = run_detection_on_frame(model, image)

        ui_frame = draw_ui(
            annotated_frame,
            danger_detected,
            detected_texts,
            max_conf,
            fps=0.0
        )

        output_path = os.path.join(OUTPUT_DIR, f"image_demo_result_{now}.jpg")
        cv2.imwrite(output_path, ui_frame)

        print(f"결과 이미지 저장 완료: {output_path}")
        print("ESC 키를 누르면 창이 종료됩니다.")

        while True:
            cv2.imshow(WINDOW_NAME, ui_frame)
            if cv2.waitKey(1) == 27:
                break

        cv2.destroyAllWindows()

    elif ext in video_exts:
        cap = cv2.VideoCapture(str(source_path))

        if not cap.isOpened():
            print("영상 파일을 열 수 없습니다.")
            return

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps == 0:
            video_fps = 30

        output_path = os.path.join(OUTPUT_DIR, f"video_demo_result_{now}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            video_fps,
            (CANVAS_WIDTH, CANVAS_HEIGHT)
        )

        print("영상 시연을 시작합니다.")
        print("ESC 키를 누르면 중단됩니다.")

        prev_time = time.time()

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            current_time = time.time()
            fps_now = 1 / (current_time - prev_time) if current_time != prev_time else 0
            prev_time = current_time

            annotated_frame, danger_detected_now, detected_texts, max_conf = run_detection_on_frame(model, frame)

            if danger_detected_now:
                last_warning_time = time.time()

            danger_detected = (time.time() - last_warning_time) <= WARNING_HOLD_SECONDS

            ui_frame = draw_ui(
                annotated_frame,
                danger_detected,
                detected_texts,
                max_conf,
                fps=fps_now
            )

            out.write(ui_frame)
            cv2.imshow(WINDOW_NAME, ui_frame)

            if cv2.waitKey(1) == 27:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print(f"결과 영상 저장 완료: {output_path}")

    else:
        print("지원하지 않는 파일 형식입니다.")
        print("이미지 또는 영상 파일을 선택해주세요.")


if __name__ == "__main__":
    main()
