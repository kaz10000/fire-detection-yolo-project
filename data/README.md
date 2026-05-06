# Data Folder Guide / 데이터 폴더 안내


이 폴더는 **스마트폰 기반 실시간 화재 감지 AI 시스템** 프로젝트의 데이터셋을 관리하는 공간입니다.

본 프로젝트는 YOLO 모델을 활용하여 카메라 영상에서 **화재(fire)**를 실시간으로 감지하는 것을 목표로 합니다.

---------------------------------------------------------------------------------------------------------------------------------------------------------------

## Folder Structure / 폴더 구조

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

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Folder Description / 폴더 세부 설명

- `raw/`  
  수집한 원본 이미지를 보관하는 폴더입니다.  
  This folder stores the original collected images.

- `raw/fire/`  
  화재 또는 불꽃이 포함된 원본 이미지를 저장합니다.  
  This folder stores original images that contain fire or flames.

- `raw/normal/`  
  화재가 없는 일반 배경 이미지를 저장합니다.  
  This folder stores normal background images without fire.

- `processed/`  
  YOLO 학습에 사용할 수 있도록 정리된 데이터를 저장합니다.  
  This folder stores the dataset organized for YOLO training.

- `processed/train/`  
  모델이 학습할 때 사용하는 데이터입니다.  
  This data is used for training the model.

- `processed/valid/`  
  학습 중 모델 성능을 확인하는 데이터입니다.  
  This data is used to validate the model during training.

- `processed/test/`  
  학습이 끝난 뒤 최종 성능을 확인하는 데이터입니다.  
  This data is used to test the final model after training.

- `images/`  
  이미지 파일을 저장하는 폴더입니다.  
  This folder stores image files.

- `labels/`  
  YOLO 형식의 라벨 파일을 저장하는 폴더입니다.  
  This folder stores YOLO-format label files.

- `dataset_sources.md`  
  데이터 출처, 수집 개수, 담당자 등을 기록하는 문서입니다.  
  This file records dataset sources, image counts, and collectors.

- `data.yaml`  
  YOLO 모델에게 학습 데이터 위치와 클래스 이름을 알려주는 설정 파일입니다.  
  This file tells YOLO where the dataset is located and what class names are used.
