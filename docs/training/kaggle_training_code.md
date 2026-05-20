# Kaggle YOLOv8n Training Code

> Kaggle Notebook에서 Roboflow Fire-Smoke 데이터셋을 불러와 YOLOv8n 모델을 학습한 코드입니다.

---

## 1. GPU 확인

```python
!nvidia-smi
```

---

## 2. 라이브러리 설치

```python
!pip install ultralytics roboflow -q
```

---

## 3. 라이브러리 불러오기

```python
from ultralytics import YOLO
from roboflow import Roboflow
import os
import glob
```

---

## 4. Roboflow 데이터셋 다운로드

> 주의: `api_key`에는 본인의 Roboflow API Key를 입력해야 합니다.  
> GitHub에는 실제 API Key를 올리지 않습니다.

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("YOUR_WORKSPACE_NAME").project("fire-smoke-detection-odvk6")
version = project.version(1)
dataset = version.download("yolov8")
```

실제로 사용했던 형태는 아래와 같습니다.

```python
# Example format only
# rf = Roboflow(api_key="YOUR_API_KEY")
# project = rf.workspace("my-space-3zzwr").project("fire-smoke-detection-odvk6")
# version = project.version(1)
# dataset = version.download("yolov8")
```

---

## 5. 데이터셋 폴더 확인

```python
!ls
```

데이터셋 폴더가 아래처럼 생성됩니다.

```text
Fire-Smoke-Detection-1
```

---

## 6. 데이터셋 경로 설정

```python
DATASET_PATH = "/kaggle/working/Fire-Smoke-Detection-1"
```

---

## 7. 폴더 구조 확인

```python
import os

print(os.listdir(DATASET_PATH))

for split in ["train", "valid", "test"]:
    print(split, os.listdir(f"{DATASET_PATH}/{split}"))
```

정상 구조 예시:

```text
['train', 'valid', 'test', 'data.yaml']
train ['images', 'labels']
valid ['images', 'labels']
test ['images', 'labels']
```

---

## 8. 이미지와 라벨 개수 확인

```python
import glob

for split in ["train", "valid", "test"]:
    image_count = len(glob.glob(f"{DATASET_PATH}/{split}/images/*"))
    label_count = len(glob.glob(f"{DATASET_PATH}/{split}/labels/*.txt"))

    print(split)
    print("images:", image_count)
    print("labels:", label_count)
    print()
```

---

## 9. data.yaml 확인

```python
!cat {DATASET_PATH}/data.yaml
```

---

## 10. data.yaml 수정

본 프로젝트는 화재 감지를 중심으로 진행하지만, 데이터셋에 `fire`와 `smoke` 클래스가 함께 포함되어 있어 두 클래스를 사용했습니다.  
`smoke`는 화재 위험을 나타내는 보조 신호로 활용했습니다.

```python
yaml_content = f"""
path: {DATASET_PATH}
train: train/images
val: valid/images
test: test/images

names:
  0: fire
  1: smoke
"""

with open(f"{DATASET_PATH}/data.yaml", "w", encoding="utf-8") as f:
    f.write(yaml_content)

print("data.yaml 수정 완료")
```

수정 확인:

```python
!cat {DATASET_PATH}/data.yaml
```

---

## 11. 라벨 오류 확인

YOLO 라벨 형식은 아래와 같아야 합니다.

```text
class_id x_center y_center width height
```

좌표값은 0~1 사이여야 합니다.

```python
import glob

error_labels = []

for split in ["train", "valid", "test"]:
    label_files = glob.glob(f"{DATASET_PATH}/{split}/labels/*.txt")

    for label_path in label_files:
        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()

            if len(parts) != 5:
                error_labels.append((label_path, "format error", line))
                continue

            cls, x, y, w, h = parts

            try:
                values = [float(x), float(y), float(w), float(h)]
                if not all(0 <= v <= 1 for v in values):
                    error_labels.append((label_path, "coordinate range error", line))
            except:
                error_labels.append((label_path, "number conversion error", line))

print("label error count:", len(error_labels))

if error_labels:
    print(error_labels[:10])
```

---

## 12. 샘플 이미지 확인

```python
from IPython.display import Image, display
import glob

sample_images = glob.glob(f"{DATASET_PATH}/train/images/*")

print("sample image count:", len(sample_images))

if sample_images:
    display(Image(filename=sample_images[0]))
```

---

## 13. YOLOv8n 1차 학습

처음에는 안정성을 위해 `epochs=20`, `batch=8`로 학습했습니다.

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data=f"{DATASET_PATH}/data.yaml",
    epochs=20,
    imgsz=640,
    batch=8,
    patience=5,
    cache=False,
    workers=2,
    device=0,
    name="fire_detection_yolov8n_kaggle"
)
```

---

## 14. 중단 후 이어 학습

학습이 중간에 끊겼을 경우 `last.pt`를 이용하여 이어서 학습했습니다.

```python
from ultralytics import YOLO

model = YOLO("/kaggle/working/runs/detect/fire_detection_yolov8n_kaggle/weights/last.pt")

model.train(
    data=f"{DATASET_PATH}/data.yaml",
    epochs=10,
    imgsz=640,
    batch=4,
    patience=3,
    cache=False,
    workers=1,
    device=0,
    name="fire_detection_yolov8n_resume_safe"
)
```

---

## 15. 학습된 모델 확인

```python
!ls /kaggle/working/runs/detect/fire_detection_yolov8n_resume_safe/weights
```

정상 출력 예시:

```text
best.pt  last.pt
```

---

## 16. 모델 평가

```python
from ultralytics import YOLO

model = YOLO("/kaggle/working/runs/detect/fire_detection_yolov8n_resume_safe/weights/best.pt")

metrics = model.val(
    data=f"{DATASET_PATH}/data.yaml",
    imgsz=640,
    batch=8,
    device=0
)
```

---

## 17. 평가 결과

Kaggle Notebook에서 확인한 주요 성능은 아래와 같습니다.

| Metric | Result |
|---|---:|
| Precision | 약 0.9779 |
| Recall | 약 0.9687 |
| mAP@0.5 | 약 0.9911 |
| mAP@0.5:0.95 | 약 0.9301 |

---

## 18. 테스트 이미지 예측

```python
from ultralytics import YOLO

model = YOLO("/kaggle/working/runs/detect/fire_detection_yolov8n_resume_safe/weights/best.pt")

results = model.predict(
    source=f"{DATASET_PATH}/test/images",
    conf=0.5,
    save=True
)
```

---

## 19. 결과 이미지 확인

```python
from IPython.display import Image, display
import glob

result_images = glob.glob("/kaggle/working/runs/detect/predict/*.jpg")

print("result image count:", len(result_images))

if result_images:
    display(Image(filename=result_images[0]))
```

---

## 20. 결과 압축

```python
!zip -r /kaggle/working/fire_detection_results.zip /kaggle/working/runs
```

압축 파일 확인:

```python
!ls /kaggle/working
```

정상 출력 예시:

```text
fire_detection_results.zip
Fire-Smoke-Detection-1
runs
yolov8n.pt
```

---

## 21. 최종 모델 위치

최종 시연에 사용한 모델 파일은 아래 경로에 있습니다.

```text
/kaggle/working/runs/detect/fire_detection_yolov8n_resume_safe/weights/best.pt
```

GitHub 프로젝트에서는 아래 위치에 저장합니다.

```text
models/best.pt
```

---

## 22. 주의사항

- Roboflow API Key는 GitHub에 업로드하지 않습니다.
- 대용량 데이터셋 ZIP 파일은 GitHub에 직접 업로드하지 않습니다.
- 데이터셋은 Google Drive 또는 Roboflow 링크로 공유합니다.
- 최종 시연에는 `best.pt` 모델을 사용합니다.
- `last.pt`는 이어 학습이 필요한 경우에만 사용합니다.

---

## 23. 한계점

검증 데이터 기준 성능은 높았지만, 실제 시연 영상에서는 아래 상황에서 감지 성능이 낮아질 수 있습니다.

| 한계점 | 설명 |
|---|---|
| 약한 불꽃 | 작은 불꽃이나 흐릿한 불꽃을 잘 감지하지 못할 수 있음 |
| 야외 연기 | 드론 시점 또는 넓게 퍼진 연기는 smoke로 잘 잡히지 않을 수 있음 |
| 테스트 환경 차이 | 학습 데이터와 실제 영상의 배경, 각도, 거리 차이가 크면 성능 저하 가능 |

---

## 24. 개선 방향

- 약한 불꽃 이미지 추가 수집
- 야외 연기 및 산불 데이터 추가 수집
- 드론 시점 화재/연기 데이터 추가
- 실패한 테스트 영상에서 프레임 추출 후 재학습
- confidence threshold 조정
