# laonCropper Final — 영수증 자동 크롭

OCR 텍스트 밀도 클러스터링 기반 영수증 자동 검출 + 크롭 엔진.

## 구조 (SoC)

```
final/
├── detection/              # 탐지 레이어 (3가지 전략)
│   ├── bbox.py             # BBox 공유 타입
│   ├── ocr.py              # OCR DensTuned (primary)
│   ├── contour.py          # 컨투어 (fallback #1)
│   └── cliff.py            # Cliff Scan (fallback #2)
├── processor.py            # 크롭 오케스트레이터 (3단 fallback)
├── server.py               # FastAPI 엔드포인트
├── cli.py                  # CLI 인터페이스
└── README.md
```

## 파이프라인

```
입력 이미지
    │
    ▼
[1차] OCR DensTuned ──── 텍스트 검출 → 크기 필터 → y가중 DBSCAN → 최대 클러스터 bbox
    │ 실패시
    ▼
[2차] ContourDetector ── Canny + 윤곽선 → 최대 사각형
    │ 실패시
    ▼
[3차] CliffScanner ───── 중앙→외곽 에너지 절벽 탐색
    │
    ▼
BBox + padding → 크롭 → 저장
```

## 사용법

### CLI
```bash
python -m final.cli input.jpg output.jpg
```

### API
```bash
python -m final.server
# POST http://localhost:8200/crop (multipart file upload)
```

### Python
```python
from final.processor import ImageProcessor

processor = ImageProcessor()
output = processor.crop("input.jpg", "output.jpg")
```

## 의존성

- easyocr (텍스트 검출)
- scikit-learn (DBSCAN 클러스터링)
- opencv-python-headless (이미지 처리)
- fastapi, uvicorn (API 서버)
