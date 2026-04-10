# 01. Plan & Design

## 배경
laonCropper는 영수증 이미지에서 배경을 제거하는 전처리 엔진.
기존 `processor.py`는 Sobel 에지 기반 "Center-Outward Cliff Scan" 방식으로
동작했으나, 복잡한 배경에서 강건성이 낮았다.

## 의사결정
- **접근**: YOLOv8n fine-tuning + `ImageProcessor`에 `mode` 파라미터로 병행 운영
- **목표 지표**: mAP@0.5 ≥ 0.90, IoU ≥ 0.85, 크롭 성공률 ≥ 95%, CPU ≤ 300ms
- **구조**: `detector.py` (YOLO), `processor.py` (mode 분기), `main.py` (API)

## 산출물
- `docs/01-plan/features/receipt-detection.plan.md`
- `docs/02-design/features/receipt-detection.design.md`

## Design Validator 검증 (86/100 → 수정 후 97/100)
- HIGH-1: `_write_output()` 명세 추가
- HIGH-2: `evaluate.py` 출력 JSON 스키마 구체화
- MED: `device="cpu"` + 스레드 안전성 문서화
- 7건 권장사항 전부 반영
