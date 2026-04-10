# src — laonCropper 소스 코드

## 구조 (SoC)

```
src/
├── detection/              # 탐지 레이어 ── BBox만 반환, 크롭 책임 없음
│   ├── bbox.py             #   BBox 공유 타입 (x1,y1,x2,y2,confidence,method)
│   ├── ocr.py              #   OCR DensTuned (primary)
│   ├── contour.py          #   ContourDetector (fallback #1)
│   └── cliff.py            #   CliffScanner (fallback #2)
├── processor.py            # 처리 레이어 ── 탐지 순서 결정 + 크롭 + I/O
├── server.py               # API 레이어 ── FastAPI, HTTP 입출력만 담당
└── cli.py                  # CLI 레이어 ── 명령행 진입점
```

## 파이프라인

```
OCR DensTuned (1차)
    │ EasyOCR text detection → 크기 필터(>5x median 제거)
    │ → y가중(0.5) DBSCAN → 최대 클러스터 bbox
    ↓ 실패시
ContourDetector (2차)
    │ Canny + adaptiveThreshold → morphology → 최대 사각형
    ↓ 실패시
CliffScanner (3차)
    │ Sobel 에지 → 중앙→외곽 에너지 절벽 탐색
    ↓
BBox + padding → 크롭 → 저장
```

## 사용법

### Python
```python
from src.processor import ImageProcessor

processor = ImageProcessor()
output = processor.crop("input.jpg", "output.jpg")
```

### API
```bash
python -m src.server
# POST http://localhost:8200/crop
```

### CLI
```bash
python -m src.cli input.jpg output.jpg
```

### Docker
```bash
docker compose up -d
```
