
---

# 3. `data/data.yaml`

```yaml
# YOLO Dataset Configuration
# YOLO 데이터셋 설정 파일

# Korean:
# 이 파일은 YOLO 모델에게 학습 데이터가 어디에 있는지 알려주는 설정 파일입니다.
# 현재 프로젝트는 화재 감지 하나만 탐지 대상으로 사용합니다.

# English:
# This file tells the YOLO model where the training data is located.
# This project uses only fire as the detection target.

path: data/processed

train: train/images
val: valid/images
test: test/images

# Class names
# 클래스 이름
names:
  0: fire