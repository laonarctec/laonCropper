# Design: 영수증 객체 탐지 (Receipt Object Detection)

- **Feature**: receipt-detection
- **Plan Ref**: `docs/01-plan/features/receipt-detection.plan.md`
- **Created**: 2026-04-09
- **Phase**: Design
- **Level**: Dynamic

---

## 0. Plan Open Questions 해소

| # | Question | Resolution |
|---|----------|------------|
| 1 | `datasets/receipts_*/labels/` 존재? | ✅ 확인: `receipts_train` 100 images / 100 labels, YOLO normalized 포맷 (`class cx cy w h`) |
| 2 | 학습 하이퍼파라미터 | `epochs=100, imgsz=640, batch=16, optimizer=auto, patience=20` (yolov8n 기준) |
| 3 | Fallback confidence 임계값 | `conf_threshold=0.5`, bbox가 없거나 최고 confidence < 0.5일 때 Cliff Scan fallback |
| 4 | `mode` default | `"cliff"` 유지 (하위호환). 학습 완료·검증 후 별도 릴리스에서 `"yolo"` 전환 검토 |

## 1. 아키텍처 개요 (Architecture)

```
┌─────────────────────────────────────────────────┐
│ FastAPI (main.py)                               │
│  POST /crop?mode=yolo|cliff                     │
└──────────────┬──────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────┐
│ ImageProcessor (processor.py)                   │
│  __init__(mode="cliff", detector=None)          │
│  auto_crop(src, dst) -> str                     │
│   ├─ mode == "yolo": detector.detect()          │
│   │   ├─ bbox 있음 → 크롭                        │
│   │   └─ 실패 → _cliff_scan_crop() fallback     │
│   └─ mode == "cliff": _cliff_scan_crop()        │
└──────────────┬──────────────────────────────────┘
               │ (mode=yolo)
               ▼
┌─────────────────────────────────────────────────┐
│ ReceiptDetector (detector.py) [NEW]             │
│  __init__(weights_path, conf=0.5, imgsz=640)    │
│  detect(image: np.ndarray) -> BBox | None       │
│  _load_model() (lazy, 싱글톤)                    │
└──────────────┬──────────────────────────────────┘
               │
               ▼
          ultralytics.YOLO("best.pt")
```

### Separation of Concerns

- **`detector.py`**: YOLO 모델 로딩 + 추론만 담당 (순수 함수형, FastAPI 비의존)
- **`processor.py`**: 크롭 전략(결정) + fallback 로직. `_cliff_scan_crop()`은 기존 알고리즘을 프라이빗 메서드로 이관
- **`main.py`**: HTTP 레이어, `mode` 쿼리 파싱 후 `ImageProcessor`에 위임
- **`train.py` (신규)**: 학습 스크립트. 런타임 코드와 분리

## 2. 컴포넌트 상세

### 2.1 `detector.py` (신규)

```python
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from ultralytics import YOLO

@dataclass
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

class ReceiptDetector:
    def __init__(
        self,
        weights_path: str = "runs/detect/train/weights/best.pt",
        conf_threshold: float = 0.5,
        imgsz: int = 640,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self._model: YOLO | None = None  # lazy load

    def _ensure_loaded(self) -> None:
        if self._model is None:
            if not self.weights_path.exists():
                raise FileNotFoundError(f"Weights not found: {self.weights_path}")
            self._model = YOLO(str(self.weights_path))

    def detect(self, image: np.ndarray) -> BBox | None:
        """가장 confidence가 높은 단일 영수증 bbox 반환. 실패시 None.

        Thread-safety: NOT reentrant. 호출측(FastAPI async handler)은
        `run_in_threadpool` 또는 `asyncio.Lock`으로 직렬화해야 한다.
        NFR-1 (CPU 300ms) 달성을 위해 device는 명시적으로 "cpu" 고정.
        """
        self._ensure_loaded()
        results = self._model.predict(
            source=image,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            device="cpu",
            verbose=False,
        )
        if not results or len(results[0].boxes) == 0:
            return None

        boxes = results[0].boxes
        best_idx = int(boxes.conf.argmax())
        xyxy = boxes.xyxy[best_idx].cpu().numpy()
        conf = float(boxes.conf[best_idx].cpu().numpy())

        return BBox(
            x1=int(xyxy[0]), y1=int(xyxy[1]),
            x2=int(xyxy[2]), y2=int(xyxy[3]),
            confidence=conf,
        )
```

**설계 결정**:
- Lazy loading: 프로세스 시작시 모델 로드 지연 (테스트 편의 + 학습 전 import 가능)
- 싱글톤은 호출측(ImageProcessor)에서 보장 (DI 친화)
- `numpy.ndarray` 입력: OpenCV 파이프라인과 직접 호환
- `BBox` dataclass: 테스트 용이성, 타입 명확

### 2.2 `processor.py` 리팩터

```python
class ImageProcessor:
    def __init__(
        self,
        mode: Literal["cliff", "yolo"] = "cliff",
        detector: ReceiptDetector | None = None,
    ) -> None:
        self.mode = mode
        self.detector = detector
        if mode == "yolo" and detector is None:
            self.detector = ReceiptDetector()

    def auto_crop(self, image_path: str, output_path: str | None = None) -> str:
        img = cv2.imread(image_path)
        if img is None:
            return image_path

        if self.mode == "yolo":
            bbox = self._safe_detect(img)
            if bbox is not None:
                result = self._crop_with_bbox(img, bbox)
            else:
                logger.info("YOLO 탐지 실패 → Cliff Scan fallback")
                result = self._cliff_scan_crop(img)
        else:
            result = self._cliff_scan_crop(img)

        return self._write_output(image_path, output_path, result)

    def _safe_detect(self, img: np.ndarray) -> BBox | None:
        try:
            return self.detector.detect(img)
        except Exception as e:
            logger.warning(f"Detector error: {e}")
            return None

    def _crop_with_bbox(self, img: np.ndarray, bbox: BBox) -> np.ndarray:
        h, w = img.shape[:2]
        pad_x = int(w * 0.02)
        pad_y = int(h * 0.02)
        x1 = max(0, bbox.x1 - pad_x)
        y1 = max(0, bbox.y1 - pad_y)
        x2 = min(w, bbox.x2 + pad_x)
        y2 = min(h, bbox.y2 + pad_y)
        return img[y1:y2, x1:x2]

    def _cliff_scan_crop(self, img: np.ndarray) -> np.ndarray:
        """기존 Center-Outward Cliff Scan 알고리즘 (로직 변경 없음)."""
        # ... 기존 auto_crop 내부 로직을 img 입력/np.ndarray 출력 형태로 이동

    def _write_output(
        self,
        image_path: str,
        output_path: str | None,
        result: np.ndarray,
    ) -> str:
        """크롭 결과를 디스크에 저장하고 저장 경로 반환.

        파일명 규칙:
        - output_path 명시 → 그대로 사용 (호출측이 결정)
        - output_path=None → `{mode_prefix}_{원본파일명}` 을 원본 디렉토리에 저장
          · mode=cliff → "true_cliff_{name}" (기존 동작 보존)
          · mode=yolo  → "yolo_{name}"
        """
        if output_path is None:
            p = Path(image_path)
            prefix = "true_cliff" if self.mode == "cliff" else "yolo"
            output_path = str(p.parent / f"{prefix}_{p.name}")
        cv2.imwrite(output_path, result)
        return output_path
```

**변경 요약**:
- 기존 `auto_crop`의 Cliff Scan 로직을 `_cliff_scan_crop(img)`로 추출
- 파일 I/O는 `_write_output`으로 분리, 파일명 규칙 명시
- `main.py`는 기존과 동일하게 `output_path`를 명시적으로 전달 → 파일명 규칙은 호출측이 결정 (하위호환)
- YOLO → fallback 경로는 예외 안전

### 2.3 `main.py` 변경

```python
from typing import Literal
from processor import ImageProcessor
from detector import ReceiptDetector

# 싱글톤: 앱 생명주기 동안 1회 로드
_detector = ReceiptDetector()  # lazy, 첫 호출시 weights 로드
_processors = {
    "cliff": ImageProcessor(mode="cliff"),
    "yolo": ImageProcessor(mode="yolo", detector=_detector),
}

@app.post("/crop")
async def crop_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    mode: Literal["cliff", "yolo"] = "cliff",
):
    ...
    processor = _processors[mode]
    final_path = processor.auto_crop(str(tmp_path), str(output_path))
    ...
```

**결정**: 두 `ImageProcessor` 인스턴스를 미리 만들어 두되, YOLO 모델은 detector의 lazy load로 첫 `mode=yolo` 요청시까지 지연 로드.

**스레드 안전성**: `ReceiptDetector`는 reentrant-safe가 아니므로, FastAPI의 async 핸들러에서 `predict()`를 호출할 때 `starlette.concurrency.run_in_threadpool`로 감싸거나 `asyncio.Lock`으로 직렬화한다. 현재 파이프라인은 요청당 1회 추론이라 락으로 충분하다.

```python
from starlette.concurrency import run_in_threadpool
# ...
final_path = await run_in_threadpool(
    processor.auto_crop, str(tmp_path), str(output_path)
)
```

### 2.4 `train.py` (신규) — 학습 스크립트

```python
"""YOLOv8n fine-tuning for receipt detection."""
from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")
    model.train(
        data="receipt_data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        patience=20,
        project="runs/detect",
        name="train",
        exist_ok=True,
    )
    metrics = model.val()
    print(f"mAP@0.5: {metrics.box.map50:.4f}")

if __name__ == "__main__":
    main()
```

### 2.5 `evaluate.py` (신규) — 두 방식 비교

```python
"""Compare cliff vs yolo crop across dataset.

입력: datasets/receipts_test/{images,labels}
출력:
  - test_results/cliff/*.jpg, test_results/yolo/*.jpg
  - test_results/evaluate_report.json
"""

# 출력 JSON 스키마
# {
#   "dataset": "receipts_test",
#   "n_images": 100,
#   "modes": {
#     "cliff": {
#       "success_rate": 0.93,        # 크롭 파일 생성 성공 비율 (NFR 검증)
#       "mean_ms": 85.2,             # 평균 처리 시간 (NFR-1: ≤300ms)
#       "mAP50": null,               # Cliff는 bbox가 없으므로 N/A
#       "iou_mean": 0.78,            # 크롭 결과 vs GT bbox IoU 평균
#       "per_image": [{"name": "receipt_0000.jpg", "ms": 83, "iou": 0.81}, ...]
#     },
#     "yolo": {
#       "success_rate": 0.98,
#       "mean_ms": 220.5,
#       "mAP50": 0.92,               # ultralytics model.val() 결과 (Plan 목표 ≥0.90)
#       "iou_mean": 0.88,
#       "per_image": [...]
#     }
#   },
#   "targets": {                     # Plan §2 Success Metrics와 직접 대조
#     "mAP50": {"target": 0.90, "actual_yolo": 0.92, "pass": true},
#     "iou_mean": {"target": 0.85, "actual_yolo": 0.88, "pass": true},
#     "success_rate": {"target": 0.95, "actual_yolo": 0.98, "pass": true},
#     "mean_ms": {"target": 300, "actual_yolo": 220.5, "pass": true}
#   }
# }

# 구현 포인트:
#   1. YOLO 모드의 mAP50는 ultralytics `YOLO.val(data="receipt_data.yaml")` 호출로 취득
#   2. IoU는 탐지된 bbox(YOLO) 또는 크롭된 영역의 외곽(Cliff) vs GT label bbox 비교
#      GT label은 datasets/receipts_test/labels/*.txt (YOLO normalized)
#   3. 시간 측정: time.perf_counter() (ms 단위)
#   4. 모든 targets.pass가 true여야 Check phase 통과
```

## 3. API 스펙

### `POST /crop`

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | multipart file | required | 입력 이미지 |
| `mode` | query enum | `cliff` | `cliff` 또는 `yolo` |

**응답**: 기존과 동일 — `FileResponse(image/jpeg)`. 실패시에도 원본 파일 경로 반환 (기존 동작 유지).

**에러**:
- `mode=yolo`인데 모델 가중치 미존재 → 500 대신 Cliff Scan fallback + 로그 경고 (NFR: 사용자 경험 유지)

### 3.1 Error Matrix

| 케이스 | 발생 지점 | 처리 정책 | HTTP 상태 |
|--------|-----------|-----------|-----------|
| 잘못된 `mode` 값 (e.g. "abc") | FastAPI `Literal` 검증 | FastAPI 기본 422 | 422 |
| 업로드 파일 없음/빈 파일 | `UploadFile` 읽기 | FastAPI 기본 422 | 422 |
| `cv2.imread` 실패 (corrupt/non-image) | `ImageProcessor.auto_crop` | 원본 경로 그대로 반환 (기존 동작 보존) | 200 |
| YOLO weights 파일 미존재 | `ReceiptDetector._ensure_loaded` | `_safe_detect`가 예외 포착 → Cliff fallback | 200 |
| YOLO 추론 중 예외 (OOM, 텐서 오류 등) | `_safe_detect` | 로그 경고 + Cliff fallback | 200 |
| bbox confidence < 0.5 또는 탐지 0개 | `ReceiptDetector.detect` | None 반환 → Cliff fallback | 200 |
| 오버사이즈 이미지 | YOLO `imgsz=640`이 내부 리사이즈 | 정상 처리 (리사이즈 후 추론, 결과 bbox는 원본 좌표) | 200 |
| `cv2.imwrite` 실패 (디스크 풀) | `_write_output` | 예외 전파 → 500 | 500 |

**원칙**: 사용자 요청의 "크롭 결과 반환"은 가능한 한 성공시켜야 하므로, 탐지 경로의 모든 실패는 Cliff Scan으로 흡수한다. I/O 수준 실패만 5xx로 노출.

## 4. 데이터 흐름 (Sequence)

```
Client              main.py          ImageProcessor      ReceiptDetector     YOLO
  │ POST /crop?mode=yolo │                  │                  │              │
  ├─────────────────────>│                  │                  │              │
  │                      │ auto_crop()      │                  │              │
  │                      ├─────────────────>│                  │              │
  │                      │                  │ detect(img)      │              │
  │                      │                  ├─────────────────>│              │
  │                      │                  │                  │ predict()    │
  │                      │                  │                  ├─────────────>│
  │                      │                  │                  │<─ boxes ─────┤
  │                      │                  │<─ BBox or None ──┤              │
  │                      │                  │                  │              │
  │                      │                  │ [None인 경우]     │              │
  │                      │                  │ _cliff_scan_crop │              │
  │                      │                  │                  │              │
  │                      │<── output path ──┤                  │              │
  │<── FileResponse ─────┤                  │                  │              │
```

## 5. 파일 변경 목록

| 파일 | 유형 | 설명 |
|------|------|------|
| `detector.py` | 신규 | `ReceiptDetector`, `BBox` |
| `processor.py` | 수정 | `mode` 파라미터, `_cliff_scan_crop` 추출, YOLO 분기 |
| `main.py` | 수정 | `mode` query, detector 싱글톤, processor dict |
| `train.py` | 신규 | 학습 스크립트 |
| `evaluate.py` | 신규 | 비교 평가 스크립트 |
| `.gitignore` | 수정 | `runs/`, `*.cache` 확인 |
| `README.md` | 수정 | 학습/추론 사용법 섹션 |

## 6. 학습 파이프라인

```
데이터 확인 (100 train / N test / N eval)
        ↓
python train.py
        ↓
runs/detect/train/weights/best.pt 생성
        ↓
python evaluate.py (cliff vs yolo 비교)
        ↓
mAP, 성공률, 추론 시간 리포트 → docs/03-analysis/
```

**학습 커맨드**:
```bash
uv run python train.py
```

**재현성**:
- Seed: 기본 `0` (ultralytics 기본)
- 하이퍼파라미터는 `train.py`에 하드코딩 (변경시 diff 추적)

## 7. 테스트 전략

**프레임워크**: `pytest` (신규 의존성). `test_hf.py` / `test_all.py` 는 기존 ad-hoc 스크립트라 건드리지 않고, 신규 구조화 테스트는 `tests/` 디렉토리에 둔다.

### Unit (`tests/`)
- `tests/test_detector.py`
  - `ReceiptDetector.detect()` — monkeypatch로 `self._model` 에 mock 주입, `BBox` 변환 로직 검증
  - weights 미존재시 `_ensure_loaded()`가 `FileNotFoundError` 발생 확인
  - confidence 0개 / 1개 / 다수 케이스
- `tests/test_processor_fallback.py`
  - `ImageProcessor(mode="yolo")` + detector가 None 반환시 Cliff 경로 진입
  - `_crop_with_bbox()` padding/clipping 경계 케이스 (이미지 가장자리 bbox)
  - `ImageProcessor()` default 생성 (= mode="cliff") → 기존 동작 보존 확인 (하위호환)
  - `_write_output` 파일명 규칙 (cliff prefix vs yolo prefix)

### Integration
- `evaluate.py`를 `receipts_test` 셋에 돌려 성공률/속도/mAP/IoU 리포트 (`evaluate_report.json` 자동 생성)
- `/crop?mode=cliff`와 `/crop?mode=yolo` 각각 샘플 1장 왕복 테스트 (httpx + TestClient)

### 성공 기준 (from Plan, evaluate_report.json의 `targets.*.pass` 전체 true)
- `mAP50 ≥ 0.90` (yolo 모드)
- `iou_mean ≥ 0.85` (yolo 모드)
- `success_rate ≥ 0.95` (yolo 모드)
- `mean_ms ≤ 300` (yolo 모드, CPU 단일 이미지)

### 하위 호환 보장 (Backwards Compatibility)

기존 사용 패턴이 변경 없이 동작해야 한다:

| 기존 호출 | 동작 보장 |
|-----------|-----------|
| `ImageProcessor()` | mode="cliff" default, 기존 알고리즘 100% 동일 |
| `processor.auto_crop(path)` | `true_cliff_{name}` 파일 생성 (파일명 유지) |
| `processor.auto_crop(path, out_path)` | `out_path`에 저장 (기존 동작 그대로) |
| `POST /crop` (mode query 없음) | mode="cliff"로 동작, FileResponse 응답 포맷 동일 |
| `test_hf.py`, `test_all.py` | 수정 불필요, 동일 결과 |

## 8. Config

런타임 튜닝 가능한 값은 생성자 기본값 + 환경변수 override 패턴으로 노출한다 (NFR-2 재현성).

| 설정 | 환경변수 | 기본값 | 설명 |
|------|----------|--------|------|
| Weights 경로 | `RECEIPT_WEIGHTS_PATH` | `runs/detect/train/weights/last.pt` | `ReceiptDetector.weights_path`. last.pt 선택 이유: best.pt는 fitness로 선택되어 precision 우선·recall 저조 케이스 발생 가능. fallback 경로 있는 우리 유스케이스는 recall 우선. |
| Confidence 임계값 | `RECEIPT_CONF_THRESHOLD` | `0.25` | fallback 트리거 기준. ultralytics 기본값과 동일. 0.5는 너무 엄격해 재현율 저하. |
| 입력 해상도 | `RECEIPT_IMGSZ` | `640` | YOLO imgsz |
| Default 모드 | `RECEIPT_MODE_DEFAULT` | `cliff` | `/crop`의 mode query default |

```python
# detector.py 예시
import os

class ReceiptDetector:
    def __init__(
        self,
        weights_path: str | None = None,
        conf_threshold: float | None = None,
        imgsz: int | None = None,
    ) -> None:
        self.weights_path = Path(
            weights_path or os.getenv("RECEIPT_WEIGHTS_PATH", "runs/detect/train/weights/best.pt")
        )
        self.conf_threshold = float(
            conf_threshold if conf_threshold is not None
            else os.getenv("RECEIPT_CONF_THRESHOLD", "0.5")
        )
        self.imgsz = int(imgsz or os.getenv("RECEIPT_IMGSZ", "640"))
        self._model = None
```

학습 하이퍼파라미터는 `train.py`에 하드코딩 유지 (실험 재현을 위해 git diff로 추적).

## 9. 리스크 & 완화 (설계 시점)

| 리스크 | 완화 |
|--------|------|
| `runs/` 디렉토리 git 오염 | `.gitignore`에 `runs/` 명시 |
| YOLO 로딩이 첫 요청 레이턴시 악화 | warm-up 엔드포인트 `/healthz` 또는 앱 시작시 명시적 `_ensure_loaded()` 호출 (선택) |
| bbox가 영수증 일부만 포함 (상단/하단 잘림) | 2% padding + fallback confidence 강화. 심한 경우 Cliff 하이브리드 v2 |
| `opencv-python-headless` + YOLO 호환성 | ultralytics 8.4.36이 headless 지원 확인됨 |
| 학습 실패/중단 | `patience=20` early stop, `runs/detect/train/` 저장으로 재개 가능 |

## 10. 구현 순서 (for Do phase)

1. `detector.py` 작성 + `BBox` 타입
2. `train.py` 작성
3. **학습 실행** → `best.pt` 산출 (수동 단계)
4. `processor.py` 리팩터 — Cliff 추출 먼저, 그 후 `mode` 분기 추가
5. `main.py` — `mode` query 및 싱글톤
6. `evaluate.py` 작성 + 실행 → 리포트
7. `.gitignore`, `README.md` 업데이트
8. 수동 스모크 테스트 (`uvicorn main:app` + 샘플 이미지)

## 11. Out of Scope (재확인)

- OCR, 필드 추출
- 다중 영수증
- GPU/양자화/ONNX (필요시 v2)

---

## Next Step

→ `/pdca do receipt-detection`
