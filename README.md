# laonCropper

영수증 이미지에서 불필요한 배경을 제거하고 종이 영역만 정밀하게 추출하는 독립형 전처리 엔진입니다.

## 🚀 설치 및 실행

### 1. 의존성 설치
```bash
uv install
```

### 2. CLI 모드 (터미널에서 직접 실행)
```bash
uv run python cli.py input.jpg [output.jpg]
```

### 3. API 모드 (서버 구동)
```bash
uv run python main.py
```
서버 실행 후 `POST http://localhost:8200/crop` 엔드포인트로 이미지를 업로드하세요.

## 🛠️ 주요 기능
- **Hybrid Edge Scan**: 이미지 중앙에서 상하좌우로 스캔하여 최적의 영수증 경계를 탐색.
- **YOLOv8 객체 탐지 (신규)**: 학습된 모델로 영수증 bbox를 직접 탐지. 실패시 Edge Scan으로 자동 fallback.
- **Perspective-ready**: 원근 변환 없이도 영수증 본체를 완벽히 보존하는 크롭 로직.
- **Zero External Dependencies**: VLM이나 외부 API 없이 OpenCV + Ultralytics만으로 고속 처리.

## 🎯 Crop 모드

| 모드 | 설명 | 사용 예 |
|------|------|---------|
| `cliff` (default) | Center-Outward Cliff Scan. 의존성 없이 동작 | `POST /crop` (모드 생략) |
| `yolo` | YOLOv8n 기반 탐지 + 실패시 cliff fallback | `POST /crop?mode=yolo` |

## 🧠 YOLOv8 학습

```bash
# 1. 데이터셋 구조 확인 (receipt_data.yaml)
#    datasets/receipts_train/{images,labels}
#    datasets/receipts_test/{images,labels}
#    datasets/receipts_eval/{images,labels}

# 2. 학습 실행 (yolov8n 기반, epochs=100, imgsz=640)
uv run python train.py
# → runs/detect/train/weights/best.pt 생성

# 3. 두 모드 비교 평가
uv run python evaluate.py
# → test_results/evaluate_report.json (mAP50, IoU, 성공률, 평균 추론 시간)
```

### 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `RECEIPT_WEIGHTS_PATH` | `runs/detect/train/weights/last.pt` | 학습된 가중치 경로 (best.pt는 fitness 선택으로 recall 저조 가능) |
| `RECEIPT_CONF_THRESHOLD` | `0.25` | Fallback 트리거 confidence 임계값 (ultralytics 기본) |
| `RECEIPT_IMGSZ` | `640` | YOLO 입력 해상도 |
| `RECEIPT_MODE_DEFAULT` | `cliff` | `/crop` endpoint의 default mode |
