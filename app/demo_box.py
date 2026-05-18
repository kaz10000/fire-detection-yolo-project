import cv2
import os
import time
from pathlib import Path
from datetime import datetime
from tkinter import Tk, filedialog

import numpy as np
from ultralytics import YOLO


# =========================
# Basic Settings
# =========================

MODEL_PATH = "models/best.pt"
OUTPUT_DIR = "results/demo_outputs"

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

CONF_THRESHOLD = 0.5
DANGER_CLASSES = ["fire", "smoke"]


# =========================
# Utility Functions
# =========================

def select_file():
    """
    Select image or video file using file dialog.
    이미지 또는 영상 파일을 선택하는 함수
    """
    root = Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select fire image or video",
        filetypes=[
            ("Image/Video files", "*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*")
        ]
    )

    root.destroy()
    return file_path


def resize_with_padding(image, target_w, target_h):
    """
    Resize image while keeping aspect ratio and add padding.
    비율을 유지하면서 이미지 크기를 맞추는 함수
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


def draw_ui(frame, danger_detected, detected_texts, max_conf, fps=0.0):
    """
    Draw demo UI.
    시연용 화면 UI를 그리는 함수
    """
    canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)

    # Background
    canvas[:] = (18, 24, 38)

    # Header
    if danger_detected:
        header_color = (0, 0, 180)
        header_text = "WARNING: FIRE RISK DETECTED"
    else:
        header_color = (0, 120, 0)
        header_text = "STATUS: SAFE"

    cv2.rectangle(canvas, (0, 0), (CANVAS_WIDTH, 90), header_color, -1)

    cv2.putText(
        canvas,
        "AI Fire Detection Safety System",
        (40, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.putText(
        canvas,
        header_text,
        (40, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # Video area border
    cv2.rectangle(
        canvas,
        (VIDEO_AREA_X - 4, VIDEO_AREA_Y - 4),
        (VIDEO_AREA_X + VIDEO_AREA_W + 4, VIDEO_AREA_Y + VIDEO_AREA_H + 4),
        (90, 120, 180),
        2
    )

    # Resize frame into video area
    display_frame = resize_with_padding(frame, VIDEO_AREA_W, VIDEO_AREA_H)
    canvas[
        VIDEO_AREA_Y:VIDEO_AREA_Y + VIDEO_AREA_H,
        VIDEO_AREA_X:VIDEO_AREA_X + VIDEO_AREA_W
    ] = display_frame

    # Right panel
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

    # Panel title
    cv2.putText(
        canvas,
        "Detection Info",
        (PANEL_X + 25, PANEL_Y + 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # Status
    status_text = "DANGER" if danger_detected else "SAFE"
    status_color = (0, 0, 255) if danger_detected else (0, 220, 0)

    cv2.putText(
        canvas,
        f"Status: {status_text}",
        (PANEL_X + 25, PANEL_Y + 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        status_color,
        2
    )

    cv2.putText(
        canvas,
        f"Max Confidence: {max_conf:.2f}",
        (PANEL_X + 25, PANEL_Y + 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (230, 230, 230),
        2
    )

    cv2.putText(
        canvas,
        f"FPS: {fps:.1f}",
        (PANEL_X + 25, PANEL_Y + 185),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (230, 230, 230),
        2
    )

    # Detected labels
    cv2.putText(
        canvas,
        "Detected:",
        (PANEL_X + 25, PANEL_Y + 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    y = PANEL_Y + 280

    if detected_texts:
        for text in detected_texts[:6]:
            cv2.putText(
                canvas,
                f"- {text}",
                (PANEL_X + 25, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1
            )
            y += 35
    else:
        cv2.putText(
            canvas,
            "- None",
            (PANEL_X + 25, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 180, 180),
            1
        )

    # Footer
    cv2.putText(
        canvas,
        "Press ESC to exit",
        (PANEL_X + 25, PANEL_Y + PANEL_H - 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (180, 180, 180),
        1
    )

    return canvas


def run_detection_on_frame(model, frame):
    """
    Run YOLO detection on one frame.
    한 프레임에 대해 YOLO 탐지를 수행하는 함수
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

        detected_texts.append(f"{class_name}: {conf:.2f}")
        max_conf = max(max_conf, conf)

        if class_name in DANGER_CLASSES:
            danger_detected = True

    return annotated_frame, danger_detected, detected_texts, max_conf


# =========================
# Main Demo Logic
# =========================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        print("Please place best.pt in the models folder.")
        return

    print("Loading model...")
    model = YOLO(MODEL_PATH)

    source_path = select_file()

    if not source_path:
        print("No file selected.")
        return

    source_path = Path(source_path)
    ext = source_path.suffix.lower()

    print(f"Selected file: {source_path}")

    image_exts = [".jpg", ".jpeg", ".png", ".bmp"]
    video_exts = [".mp4", ".avi", ".mov", ".mkv"]

    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Image demo
    if ext in image_exts:
        image = cv2.imread(str(source_path))

        if image is None:
            print("Cannot read image file.")
            return

        annotated_frame, danger_detected, detected_texts, max_conf = run_detection_on_frame(model, image)
        ui_frame = draw_ui(annotated_frame, danger_detected, detected_texts, max_conf)

        output_path = os.path.join(OUTPUT_DIR, f"image_demo_result_{now}.jpg")
        cv2.imwrite(output_path, ui_frame)

        print(f"Result saved: {output_path}")
        print("Press ESC to close the window.")

        while True:
            cv2.imshow(WINDOW_NAME, ui_frame)
            if cv2.waitKey(1) == 27:
                break

        cv2.destroyAllWindows()

    # Video demo
    elif ext in video_exts:
        cap = cv2.VideoCapture(str(source_path))

        if not cap.isOpened():
            print("Cannot open video file.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30

        output_path = os.path.join(OUTPUT_DIR, f"video_demo_result_{now}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (CANVAS_WIDTH, CANVAS_HEIGHT))

        print("Video demo started.")
        print("Press ESC to stop.")

        prev_time = time.time()

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            current_time = time.time()
            fps_now = 1 / (current_time - prev_time) if current_time != prev_time else 0
            prev_time = current_time

            annotated_frame, danger_detected, detected_texts, max_conf = run_detection_on_frame(model, frame)
            ui_frame = draw_ui(annotated_frame, danger_detected, detected_texts, max_conf, fps=fps_now)

            out.write(ui_frame)
            cv2.imshow(WINDOW_NAME, ui_frame)

            if cv2.waitKey(1) == 27:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print(f"Result video saved: {output_path}")

    else:
        print("Unsupported file format.")
        print("Please select an image or video file.")


if __name__ == "__main__":
    main()
