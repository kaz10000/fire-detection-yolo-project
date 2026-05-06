# Smartphone-Based Real-Time Fire Detection AI System  
# 스마트폰 기반 실시간 화재 감지 AI 시스템

## 1. Project Overview / 프로젝트 개요

본 프로젝트는 스마트폰 또는 웹캠 카메라 영상을 활용하여 **화재 상황을 실시간으로 감지하는 AI 시스템**을 개발하는 것을 목표로 합니다.  
YOLO 모델을 이용해 영상 속 화재 객체를 탐지하고, 화재가 감지되면 화면에 경고 메시지를 표시합니다.

This project aims to develop an **AI system that detects fire in real time** using a smartphone or webcam camera.  
The system uses a YOLO model to detect fire objects in video frames and displays a warning message when fire is detected.

---

## 2. Project Folder Structure / 프로젝트 폴더 구조

```text
fire-detection-yolo-project/           # 프로젝트 최상위 폴더 / Main project folder
├─ app/                                # 실행 코드 저장 폴더 / Source code folder
├─ data/                               # 데이터셋 관리 폴더 / Dataset management folder
├─ docs/                               # 문서 및 발표 자료 폴더 / Documents and presentation folder
├─ models/                             # 학습된 모델 저장 폴더 / Trained model folder
├─ results/                            # 테스트 결과 저장 폴더 / Test result folder
├─ README.md                           # 프로젝트 전체 설명 파일 / Main project guide file
└─ .gitignore                          # GitHub 업로드 제외 설정 파일 / Git ignore setting file
```

---

## 3. Folder Description / 폴더 설명

| Folder | Description |
|---|---|
| `app/` | 파이썬 실행 코드와 학습/테스트 코드를 저장하는 폴더 / Stores Python source code for training, testing, and real-time demo |
| `data/` | 화재 이미지와 일반 이미지 데이터셋을 관리하는 폴더 / Manages fire and normal image datasets |
| `docs/` | 제안서, 발표 PPT, 회의록 등을 저장하는 폴더 / Stores proposal, presentation slides, and meeting notes |
| `models/` | YOLO 사전학습 모델 또는 학습 완료 모델을 저장하는 폴더 / Stores pretrained YOLO models and trained model weights |
| `results/` | 탐지 결과 이미지, 영상, 그래프 등을 저장하는 폴더 / Stores detection screenshots, videos, and result graphs |

---

## 4. Detailed Folder Usage / 세부 폴더 사용법

### 4.1 `app/` — Code Folder / 코드 폴더

```text
app/                                   # 코드 저장 폴더 / Code folder
├─ camera_test.py                      # 카메라 연결 테스트 코드 / Camera connection test code
├─ train_yolo.py                       # YOLO 모델 학습 코드 / YOLO training code
├─ predict_image.py                    # 이미지 1장 예측 테스트 코드 / Single image prediction test code
├─ realtime_demo.py                    # 실시간 화재 감지 실행 코드 / Real-time fire detection demo code
└─ utils.py                            # 공통 기능 함수 저장 파일 / Utility functions
```

`app/` 폴더에는 프로젝트 실행에 필요한 파이썬 코드를 저장합니다.  
처음에는 카메라가 잘 연결되는지 확인하기 위해 `camera_test.py`를 실행하고, 이후 YOLO 학습과 실시간 감지 코드를 추가합니다.

The `app/` folder stores Python code required to run the project.  
First, use `camera_test.py` to check whether the camera works properly. Then add YOLO training and real-time detection code.

---

### 4.2 `data/` — Dataset Folder / 데이터셋 폴더

```text
data/                                  # 데이터셋 전체 관리 폴더 / Main dataset folder
├─ raw/                                # 원본 데이터 저장 폴더 / Original raw data folder
│  ├─ fire/                            # 수집한 화재 이미지 원본 저장 / Original fire images
│  └─ normal/                          # 화재가 없는 일반 이미지 원본 저장 / Original normal images without fire
│
├─ processed/                          # YOLO 학습용으로 정리된 데이터 폴더 / Processed dataset for YOLO training
│  ├─ train/                           # 모델 학습용 데이터 / Training dataset
│  │  ├─ images/                       # 학습용 이미지 파일 / Training image files
│  │  └─ labels/                       # 학습용 YOLO 라벨 파일 / Training YOLO label files
│  ├─ valid/                           # 모델 검증용 데이터 / Validation dataset
│  │  ├─ images/                       # 검증용 이미지 파일 / Validation image files
│  │  └─ labels/                       # 검증용 YOLO 라벨 파일 / Validation YOLO label files
│  └─ test/                            # 최종 테스트용 데이터 / Final testing dataset
│     ├─ images/                       # 테스트용 이미지 파일 / Test image files
│     └─ labels/                       # 테스트용 YOLO 라벨 파일 / Test YOLO label files
│
├─ README.md                           # 데이터 폴더 사용 설명서 / Data folder guide
├─ dataset_sources.md                  # 데이터 출처 및 수집 기록 문서 / Dataset source and collection log
└─ data.yaml                           # YOLO 학습 설정 파일 / YOLO dataset configuration file
```

`data/` 폴더는 프로젝트에서 사용하는 모든 데이터를 관리합니다.  
원본 이미지는 `raw/`에 저장하고, YOLO 학습에 맞게 정리한 이미지는 `processed/`에 저장합니다.  
화재 감지 프로젝트이므로 기본 데이터는 `fire`와 `normal`로 나눕니다.

The `data/` folder manages all datasets used in the project.  
Original images are stored in `raw/`, and YOLO-ready training data is stored in `processed/`.  
Since this is a fire detection project, the data is divided into `fire` and `normal`.

---

### 4.3 `docs/` — Documents Folder / 문서 폴더

```text
docs/                                  # 문서 저장 폴더 / Documents folder
├─ proposal/                           # 프로젝트 제안서 저장 / Project proposal files
├─ ppt/                                # 발표 PPT 저장 / Presentation slides
├─ meeting_notes/                      # 회의록 저장 / Meeting notes
└─ references/                         # 참고 자료 저장 / Reference materials
```

`docs/` 폴더에는 제안서, 발표 자료, 회의록, 참고 자료를 저장합니다.  
교수님께 제출하는 문서나 팀원과 공유할 자료는 이 폴더에 정리합니다.

The `docs/` folder stores proposals, presentation slides, meeting notes, and references.  
Documents submitted to the professor or shared with team members should be organized here.

---

### 4.4 `models/` — Model Folder / 모델 폴더

```text
models/                                # 모델 저장 폴더 / Model storage folder
├─ yolov8n.pt                          # YOLOv8 사전학습 모델 / Pretrained YOLOv8 model
├─ best.pt                             # 학습 후 가장 성능이 좋은 모델 / Best trained model
└─ README.md                           # 모델 사용 설명 / Model usage guide
```

`models/` 폴더에는 YOLO 모델 파일을 저장합니다.  
처음에는 `yolov8n.pt` 같은 사전학습 모델을 사용하고, 직접 학습 후 생성되는 `best.pt` 파일을 이곳에 저장합니다.

The `models/` folder stores YOLO model files.  
At first, use a pretrained model such as `yolov8n.pt`. After custom training, save the best trained model file, such as `best.pt`, in this folder.

---

### 4.5 `results/` — Result Folder / 결과 폴더

```text
results/                               # 결과 저장 폴더 / Result folder
├─ screenshots/                        # 탐지 결과 스크린샷 저장 / Detection screenshots
├─ videos/                             # 탐지 결과 영상 저장 / Detection result videos
├─ graphs/                             # 성능 그래프 저장 / Performance graphs
└─ reports/                            # 테스트 결과 보고서 저장 / Test result reports
```

`results/` 폴더에는 모델 실행 결과를 저장합니다.  
실시간 감지 화면 캡처, 결과 영상, 성능 그래프, 테스트 결과 표 등을 이곳에 정리합니다.  
PPT나 제안서에 들어갈 결과 자료도 이 폴더에서 관리합니다.

The `results/` folder stores model execution results.  
Detection screenshots, videos, performance graphs, and test result tables should be organized here.  
Materials used in the PPT or proposal can also be managed in this folder.

---

## 5. Dataset Rule / 데이터셋 규칙

본 프로젝트는 화재 감지만을 메인 기능으로 진행합니다.

- 탐지 클래스: `fire`
- 일반 이미지: `normal`
- `normal` 이미지는 화재가 없는 배경 이미지로 사용합니다.
- 전체 데이터셋은 GitHub에 직접 올리지 않고, Google Drive 또는 Roboflow 링크로 공유하는 것을 권장합니다.
- GitHub에는 샘플 이미지, 데이터 설명 파일, 학습 코드 위주로 업로드합니다.

This project focuses only on fire detection.

- Detection class: `fire`
- Normal images: `normal`
- `normal` images are used as background images without fire.
- It is recommended not to upload the entire dataset directly to GitHub. Use Google Drive or Roboflow links instead.
- Upload sample images, dataset guide files, and source code to GitHub.

---

## 6. Recommended Dataset Size / 권장 데이터 수

| Class | Description | Recommended Count |
|---|---|---:|
| `fire` | 화재 이미지 / Fire images | 500~800 |
| `normal` | 일반 배경 이미지 / Normal images | 500~800 |
| Total | 전체 이미지 / Total images | 1,000~1,600 |

### Minimum Version / 최소 기준

```text
fire: 300 images
normal: 300 images
total: 600 images
```

---

## 7. Data Split Rule / 데이터 분할 기준

```text
train: 70%
valid: 20%
test: 10%
```

| Type | Description | Ratio |
|---|---|---:|
| train | 학습용 데이터 / Training data | 70% |
| valid | 검증용 데이터 / Validation data | 20% |
| test | 테스트용 데이터 / Testing data | 10% |

---

## 8. How to Run / 실행 방법

### 8.1 Install Libraries / 라이브러리 설치

```bash
pip install ultralytics opencv-python numpy pandas matplotlib
```

### 8.2 Camera Test / 카메라 테스트

```bash
python app/camera_test.py
```

### 8.3 Train YOLO Model / YOLO 모델 학습

```bash
python app/train_yolo.py
```

### 8.4 Run Real-Time Demo / 실시간 데모 실행

```bash
python app/realtime_demo.py
```

---

## 9. GitHub Collaboration Rule / GitHub 협업 규칙

작업 전에는 항상 최신 파일을 받아옵니다.

Before working, always pull the latest files.

```bash
git pull origin main
```

작업 후에는 아래 순서로 업로드합니다.

After working, upload your changes using the following commands.

```bash
git add .
git commit -m "work description"
git pull --rebase origin main
git push origin main
```

---

## 10. Commit Message Examples / 커밋 메시지 예시

```text
[docs] Add project README
[data] Add fire dataset guide
[app] Add camera test code
[model] Add YOLO training script
[result] Add detection screenshots
[fix] Fix README formatting
```

---

## 11. Upload Rule / 업로드 규칙

GitHub에 올려도 되는 것:

Allowed to upload to GitHub:

- 코드 파일 / Source code
- README 문서 / README files
- 데이터 출처 문서 / Dataset source documents
- 샘플 이미지 몇 장 / A few sample images
- 결과 스크린샷 / Result screenshots
- 제안서, PPT / Proposal and PPT files

GitHub에 직접 올리지 않는 것이 좋은 것:

Not recommended to upload directly to GitHub:

- 대용량 데이터셋 전체 / Large full datasets
- 큰 영상 파일 / Large video files
- 개인정보가 포함된 이미지 / Images containing personal information
- 저작권이 불분명한 이미지 / Images with unclear copyright
- 너무 큰 모델 파일 / Very large model files

---

## 12. Project Progress Plan / 프로젝트 진행 계획

| Step | Task |
|---|---|
| 1 | 프로젝트 범위 확정 / Finalize project scope |
| 2 | 데이터셋 수집 / Collect dataset |
| 3 | 데이터 라벨링 및 정리 / Label and organize data |
| 4 | YOLO 모델 학습 / Train YOLO model |
| 5 | 실시간 카메라 연결 / Connect real-time camera |
| 6 | 화재 감지 경고 UI 구현 / Build fire warning UI |
| 7 | 테스트 및 결과 정리 / Test and organize results |
| 8 | 제안서 및 PPT 완성 / Complete proposal and PPT |
| 9 | 최종 발표 시연 준비 / Prepare final demo |

---

## 13. Final Goal / 최종 목표

최종 목표는 스마트폰 또는 웹캠 영상을 입력받아 YOLO 모델로 화재 상황을 실시간 감지하고, 화재가 감지되면 화면에 경고 메시지를 표시하는 시스템을 구현하는 것입니다.

The final goal is to build a system that receives smartphone or webcam video, detects fire in real time using a YOLO model, and displays a warning message when fire is detected.
