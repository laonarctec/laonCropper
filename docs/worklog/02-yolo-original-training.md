# 02. YOLO 원본 학습 (100장)

## 구현
- `detector.py`: ReceiptDetector + BBox dataclass, lazy load, device="cpu"
- `train.py`: YOLOv8n fine-tuning (epochs=100, imgsz=640, patience=20, seed=0)
- `evaluate.py`: cliff vs yolo 비교, JSON 리포트 + targets 검증

## 학습 결과
- GPU OOM → CPU 전환 (공유 GPU에 llama-server 20GB 점유)
- mAP@0.5: **0.9725**, Early stop at epoch 23 (best epoch 3)
- best.pt의 P=1.0/R=0.639 문제 발견 → **last.pt + conf=0.25**로 변경

## 평가 결과 (원본 test set)
- mAP@0.5: 0.9725, IoU: 0.8573, success: 100%, mean: 28ms
- Success Metrics: ✅ PASS

## 실샘플 테스트 결과 — ❌ 실패
**근본 원인**: 학습 데이터 라벨의 평균 면적 = 이미지의 **86.3%**.
100장 모두 "영수증이 이미지 대부분"인 구도 → 모델이 "이미지 전체 반환"을 학습.
배경이 큰 실사진에서는 bbox가 95%~100% → 의미 없는 크롭.
