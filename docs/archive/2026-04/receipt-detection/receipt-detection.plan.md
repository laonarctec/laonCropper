# Plan: 영수증 객체 탐지 (Receipt Object Detection)

- **Feature**: receipt-detection
- **Created**: 2026-04-09
- **Owner**: TreeAnderson
- **Phase**: Plan
- **Level**: Dynamic

---

## 1. 배경 (Background)

현재 `processor.py`의 `ImageProcessor.auto_crop()`은 Sobel 에지 기반 "Center-Outward Cliff Scan" 방식으로 영수증 영역을 찾는다. 이 방식은 아래와 같은 한계가 있다.

- 복잡한 배경(무늬 있는 테이블, 타일 등)에서 에지 노이즈가 본문 경계와 혼동됨
- 임계값(본문 평균 × 0.20)과 고정 마진(좌 2%, 우 6%, 상하 4%)이 휴리스틱이라 강건성이 낮음
- 이미지 내 영수증 비율이 85%를 넘지 않는다는 가정이 고정 하드코딩되어 있음
- `test_results_hf/`, `test_hf.py` 기반 100종 데이터셋 테스트에서 실패 케이스가 관찰됨

프로젝트에는 이미 `receipt_data.yaml` (YOLO 데이터셋 설정), `datasets/receipts_{train,test,eval}/`, `yolov8n.pt` (사전학습 가중치)가 준비되어 있어, 학습 기반 객체 탐지로 전환할 수 있는 기반이 갖춰져 있다.

## 2. 목표 (Goals)

YOLOv8 기반 영수증 전용 탐지기를 학습 및 통합하여, 기존 Cliff Scan 대비 **정확도·강건성**을 개선하고, 기존 API는 깨지지 않도록 **옵션으로 병행 운영**한다.

### Success Metrics

| 지표 | 목표 | 측정 방법 |
|------|------|-----------|
| mAP@0.5 | ≥ 0.90 | `receipts_test` 셋 기준 YOLO val |
| IoU (mean) | ≥ 0.85 | 테스트 셋 GT bbox 대비 |
| 크롭 성공률 | ≥ 95% | `test_hf.py` 스타일 스크립트 |
| CPU 추론 속도 | ≤ 300ms/img (yolov8n, 640px) | 단일 스레드 기준 |
| 강건성 | 복잡 배경 10종에서 성공률 90% 이상 | 수동 큐레이션 셋 |

## 3. 비목표 (Non-Goals)

- OCR, 필드 추출(금액/날짜)은 이 기능의 범위 밖
- 다중 영수증 탐지 → v1에서는 단일 영수증(가장 큰 bbox)만 처리
- GPU 추론 최적화 → CPU 기준이 우선
- 기존 Cliff Scan 알고리즘 제거 → fallback으로 유지

## 4. 요구사항 (Requirements)

### Functional

- **FR-1**: YOLOv8n을 `receipt_data.yaml` 기반으로 fine-tuning 가능해야 한다.
- **FR-2**: 학습된 가중치(`runs/detect/train/weights/best.pt`)를 로드하는 `ReceiptDetector` 클래스를 제공한다.
- **FR-3**: `ImageProcessor`는 생성자 옵션(`mode={"cliff","yolo"}`)으로 탐지 방식을 선택할 수 있다. 기본값은 기존 동작 보존을 위해 `"cliff"`로 둔다.
- **FR-4**: YOLO 모드에서 탐지 실패(confidence 임계값 미달)시 자동으로 Cliff Scan으로 fallback 한다.
- **FR-5**: `/crop` API는 선택적 query 파라미터 `mode`를 받아 방식을 지정할 수 있다.
- **FR-6**: 평가 스크립트(`test_all.py` 확장 또는 신규)로 두 방식을 같은 데이터셋에서 비교 가능해야 한다.

### Non-Functional

- **NFR-1**: 추론은 CPU에서 단일 이미지 300ms 이하
- **NFR-2**: 학습 재현 가능성 → 학습 커맨드/하이퍼파라미터를 문서화
- **NFR-3**: 메모리 — 모델 로딩은 프로세스당 1회(싱글톤)
- **NFR-4**: 기존 `/crop` 엔드포인트의 default 응답 포맷(파일 다운로드)은 변경하지 않음

## 5. 접근 방식 (Approach)

**선택**: YOLOv8 학습 + 옵션으로 병행 (ImageProcessor에 mode 파라미터 추가)

### Rationale

- 데이터셋과 사전학습 가중치가 이미 존재 → 초기 투자비용이 낮음
- 단일 클래스(`receipt`) 탐지라 경량 모델(yolov8n)로 충분
- HuggingFace DETR 등 일반 목적 모델은 영수증 도메인 특화 정확도에서 밀림
- 기존 Cliff Scan은 fallback으로 두어 탐지 실패시 안전망 역할

### Alternatives Considered

| 대안 | Pros | Cons | 채택 여부 |
|------|------|------|-----------|
| HuggingFace 사전학습 DETR | 학습 불필요 | 영수증 특화 정확도 부족, 모델 크기 큼 | ❌ |
| YOLO + Cliff Scan 하이브리드 (bbox 후 Cliff 보정) | 경계 정밀도 극대화 | 구현 복잡도, 속도 저하 | ❌ (v2 고려) |
| 기존 Cliff Scan 파라미터 튜닝 | 의존성 변경 없음 | 강건성 한계는 구조적 | ❌ |

## 6. 범위 (Scope)

### In Scope

1. YOLOv8n fine-tuning 파이프라인 (학습 스크립트 + 하이퍼파라미터)
2. `ReceiptDetector` 클래스 (`detector.py` 신규)
3. `ImageProcessor` 리팩터링 — `mode` 옵션, fallback 로직
4. `/crop` API의 `mode` query 파라미터
5. 평가/비교 스크립트 (`evaluate.py` 또는 `test_all.py` 확장)
6. 학습/추론 커맨드 README 섹션

### Out of Scope

- OCR / 필드 파싱
- 다중 영수증
- 모바일/엣지 양자화
- 데이터셋 추가 수집 및 라벨링 (기존 데이터셋 활용)

## 7. 리스크 (Risks)

| 리스크 | 영향 | 완화책 |
|--------|------|--------|
| 학습 데이터 편향 → 복잡 배경에서 일반화 실패 | 높음 | 큐레이션 셋으로 별도 검증, augmentation 강화 |
| CPU 추론 속도가 300ms 초과 | 중간 | imgsz 축소(640→512), ONNX export 고려 |
| YOLO bbox가 영수증의 일부(상단/하단 잘림) 반환 | 중간 | bbox 외곽 padding, fallback 트리거 기준 강화 |
| `runs/` 디렉토리로 인한 git 용량 증가 | 낮음 | `.gitignore`에 `runs/` 추가 확인 |

## 8. 의존성 (Dependencies)

- Python: `ultralytics` (YOLOv8), `opencv-python`, `numpy` (기존)
- 데이터: `datasets/receipts_{train,test,eval}/images` + labels (존재 여부 설계 단계에서 확인)
- 가중치: `yolov8n.pt` (존재)
- FastAPI (기존)

## 9. 일정 / 마일스톤 (Milestones)

| 단계 | 산출물 |
|------|--------|
| M1 — Design | `docs/02-design/features/receipt-detection.design.md` (클래스 다이어그램, API 스펙, 학습 파이프라인) |
| M2 — Training | fine-tuned `best.pt`, 학습 로그, mAP 리포트 |
| M3 — Implementation | `detector.py`, `processor.py` 리팩터, `/crop` API mode 지원 |
| M4 — Evaluation | 비교 스크립트 결과, 성공률 리포트 |
| M5 — Check (Gap Analysis) | `docs/03-analysis/receipt-detection.analysis.md` |
| M6 — Report | `docs/04-report/features/receipt-detection.report.md` |

## 10. Open Questions

1. `datasets/receipts_*/labels/` YOLO 포맷 라벨이 실제로 존재하는가? (Design 단계에서 확인)
2. 학습 epoch / imgsz 기본값은? (권장: epochs=100, imgsz=640, batch=16 — Design에서 확정)
3. Fallback 트리거 기준 confidence 임계값? (초안: 0.5)
4. `mode` 파라미터 default는 하위 호환을 위해 `"cliff"` 유지, 아니면 학습 완료 후 `"yolo"`로 전환?

---

## Next Step

→ `/pdca design receipt-detection`
