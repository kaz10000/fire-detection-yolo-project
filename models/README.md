# Model Folder / 모델 폴더

이 폴더는 프로젝트에서 사용하는 YOLO 모델 파일을 저장하는 공간입니다.  
This folder stores YOLO model files used in this project.

---

## Folder Purpose / 폴더 목적

`models/` 폴더에는 학습이 완료된 화재 감지 모델 파일을 저장합니다.  
The `models/` folder stores trained fire detection model files.

본 프로젝트에서는 Kaggle Notebook에서 YOLOv8n 모델을 학습시킨 후 생성된 `best.pt` 파일을 사용합니다.  
In this project, we use the `best.pt` file generated after training the YOLOv8n model in Kaggle Notebook.

---

## Main Model File / 주요 모델 파일

```text
models/
├─ best.pt      # 최종 학습된 화재 감지 모델 / Final trained fire detection model
└─ README.md    # 모델 설명 문서 / Model description file
```

---

## Model Information / 모델 정보

| Item | Description |
|---|---|
| Model Name | YOLOv8n |
| Model File | `best.pt` |
| Task | Fire detection |
| Additional Class | Smoke detection may be included as a fire-risk signal |
| Training Platform | Kaggle Notebook |
| Dataset Source | Roboflow Universe Fire-Smoke Detection Dataset |
| Input Image Size | 640 × 640 |
| Main Purpose | Detect fire in images, videos, or real-time camera input |

---

## Training Result / 학습 결과

The trained model achieved high detection performance during validation.

| Metric | Result |
|---|---:|
| Precision | 약 97.8% |
| Recall | 약 96.9% |
| mAP@0.5 | 약 99.1% |
| mAP@0.5:0.95 | 약 93.0% |

- `Precision`: AI가 화재라고 판단한 것 중 실제로 맞은 비율  
- `Recall`: 실제 화재 중 AI가 찾아낸 비율  
- `mAP@0.5`: 객체 탐지 성능을 평가하는 대표 지표  
- `mAP@0.5:0.95`: 더 엄격한 기준에서의 탐지 성능  

---

## How to Use / 사용 방법

Python 코드에서 아래와 같이 `best.pt` 모델을 불러와 사용할 수 있습니다.

```python
from ultralytics import YOLO

model = YOLO("models/best.pt")
```

이미지 예측 예시:

```python
results = model.predict(
    source="test.jpg",
    conf=0.5,
    save=True
)
```

웹캠 또는 실시간 카메라 시연에서는 `app/realtime_demo.py`에서 이 모델을 불러와 사용합니다.

---

## Important Notes / 주의사항

- `best.pt`는 학습이 완료된 최종 모델 파일입니다.
- `last.pt`는 마지막 epoch 기준 모델이며, 일반적으로 최종 시연에는 `best.pt`를 사용합니다.
- 모델 파일 용량이 100MB를 초과하면 GitHub에 직접 업로드하지 않고 Google Drive 링크로 공유하는 것을 권장합니다.
- Roboflow API Key, 개인 정보, 대용량 원본 데이터셋은 이 폴더에 업로드하지 않습니다.

---

## File Management Rule / 파일 관리 규칙

| File | Upload to GitHub? | Note |
|---|---|---|
| `best.pt` | 가능 | 100MB 이하일 경우 업로드 가능 |
| `last.pt` | 선택 | 이어 학습이 필요할 때만 보관 |
| Large model files | 비추천 | Google Drive 링크로 공유 |
| API Key files | 금지 | 절대 업로드하지 않기 |

---

## Final Model Path / 최종 모델 경로

프로젝트에서 최종 모델 파일은 아래 위치에 저장합니다.

```text
fire-detection-yolo-project/
└─ models/
   └─ best.pt
```

This `best.pt` file is the final trained model used for fire detection demos.
