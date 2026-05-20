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

MODEL_PATH = "models/team_best.pt"
OUTPUT_DIR = "results/demo_outputs"

# OpenCV 창 제목은 한글이 깨질 수 있어서 영어로 설정
WINDOW_NAME = "AI Fire Detection Safety System"

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

# 영상에서는 confidence가 낮게 나올 수 있어서 낮게 설정
CONF_THRESHOLD = 0.15

# 모델 클래스명이 대소문자 또는 flame으로 되어 있을 가능성까지 고려
DANGER_CLASSES = ["fire", "smoke", "Fire", "Smoke", "flame", "Flame"]

# 한 번 감지되면 경고를 몇 초 동안 유지할지 설정
WARNING_HOLD_SECONDS = 2.0


# =========================
# 한글 폰트 설정
# =========================

def get_korean_font(size=28):
    """
    Windows에서 한글 출력을 위한 폰트 설정
    """
    font_candidates = [
        "C:/Windows/Fonts/malgun.ttf",        # 맑은 고딕
        "C:/Windows/Fonts/malgunbd.ttf",      # 맑은 고딕 Bold
        "C:/Windows/Fonts/gulim.ttc",         # 굴림
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
    """
    OpenCV 이미지 위에 한글 텍스트를 그리는 함수
    """
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)


# =========================
# 창 설정 함수
# =========================

def prepare_window():
    """
    OpenCV 창을 크기 조절 가능하게 설정
    X 버튼 종료와 창 크기 조절을 위해 WINDOW_NORMAL 사용
    """
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, CANVAS_WIDTH, CANVAS_HEIGHT)


def should_close_window():
    """
    ESC 키 또는 X 버튼 종료 확인
    """
    key = cv2.waitKey(1)

    if key == 27:
        return True

    try:
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            return True
    except cv2.error:
        return True

    return False


# =========================
# 파일 선택
# =========================

def select_file():
    """
    이미지 또는 영상 파일 선택 창 열기
    """
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
    """
    비율을 유지하면서 이미지/영상 프레임을 지정 영역에 맞추기
    """
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
    """
    시연용 한글 UI 화면 생성
    """
    canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
    canvas[:] = (18, 24, 38)

    # 상단 상태바
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

    # 영상 영역 테두리
    cv2.rectangle(
        canvas,
        (VIDEO_AREA_X - 4, VIDEO_AREA_Y - 4),
        (VIDEO_AREA_X + VIDEO_AREA_W + 4, VIDEO_AREA_Y + VIDEO_AREA_H + 4),
        (90, 120, 180),
        2
    )

    # 영상/이미지 삽입
    display_frame = resize_with_padding(frame, VIDEO_AREA_W, VIDEO_AREA_H)

    canvas[
        VIDEO_AREA_Y:VIDEO_AREA_Y + VIDEO_AREA_H,
        VIDEO_AREA_X:VIDEO_AREA_X + VIDEO_AREA_W
    ] = display_frame

    # 오른쪽 정보 패널
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

    # 상태 표시
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
        "ESC 키 또는 X 버튼을 누르면 종료됩니다",
        (PANEL_X + 25, PANEL_Y + PANEL_H - 45),
        FONT_SMALL,
        (180, 180, 180)
    )

    return canvas


# =========================
# YOLO 탐지
# =========================

def run_detection_on_frame(model, frame):
    """
    한 프레임에 대해 YOLO 탐지 수행
    """
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

        class_name_lower = class_name.lower()

        if class_name_lower == "fire":
            display_name = "화재"
        elif class_name_lower == "smoke":
            display_name = "연기"
        elif class_name_lower == "flame":
            display_name = "불꽃"
        else:
            display_name = class_name

        detected_texts.append(f"{display_name}: {conf:.2f}")
        max_conf = max(max_conf, conf)

        if class_name in DANGER_CLASSES or class_name_lower in ["fire", "smoke", "flame"]:
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

    # =========================
    # 이미지 시연
    # =========================

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
        print("ESC 키 또는 X 버튼을 누르면 창이 종료됩니다.")

        prepare_window()

        while True:
            cv2.imshow(WINDOW_NAME, ui_frame)

            if should_close_window():
                break

        cv2.destroyAllWindows()

    # =========================
    # 영상 시연
    # =========================

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
        print("ESC 키 또는 X 버튼을 누르면 중단됩니다.")

        prepare_window()

        prev_time = time.time()

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            current_time = time.time()
            fps_now = 1 / (current_time - prev_time) if current_time != prev_time else 0
            prev_time = current_time

            annotated_frame, danger_detected_now, detected_texts, max_conf = run_detection_on_frame(model, frame)

            # 한 번 감지되면 일정 시간 동안 경고 유지
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

            if should_close_window():
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
