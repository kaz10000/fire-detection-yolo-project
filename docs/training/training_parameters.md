# Training Parameters

> Kaggle Notebook에서 YOLOv8n 모델을 학습할 때 실제로 사용한 파라미터 정리

---

## Model Information

| Item | Value |
|---|---|
| Model | YOLOv8n |
| Training Method | Transfer Learning |
| Base Weight | `yolov8n.pt` |
| Final Model | `best.pt` |
| Training Environment | Kaggle Notebook |
| Dataset | Roboflow Fire-Smoke Detection Dataset |

---

## Actual Training Parameters

| Parameter | Value |
|---|---|
| Image Size | 640 × 640 |
| Initial Epochs | 20 |
| Resumed Epochs | 10 |
| Batch Size | 4 |
| Patience | 3 |
| Cache | False |
| Workers | 1 |
| Device | GPU |
| Demo Confidence Threshold | 0.15 |

---

## Training Code

```python
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
