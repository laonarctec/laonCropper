# 04. 라인 강조 + 증강 v2

## 접근
1. **preprocessing.py**: 4방향 Sobel 라인 강조 → 원본에 블렌딩 (alpha=0.35)
2. **augment_dataset_v2.py**: 5위치(center+N/S/E/W) × 2배경(diverse+source-edge) = ×10
3. 배경: 부드러운 그라데이션 + 강한 Gaussian blur

## 학습 결과
- 1000장 (train), mAP@0.5: **0.9911**, mAP@0.5:0.95: **0.8931**
- P=0.988, R=0.989, Early stop epoch 65 (best epoch 45)

## 실샘플 테스트 — 개선되었으나 여전히 불만족
- 6/6 탐지 성공
- YOLO 추론 ~300ms (전처리 포함)
- 사용자 평가: "111608만 괜찮고 나머지는 사용하기 어렵다"
- **과적합 재확인**: Overfitting Monitor에서 Box Loss gap 발산 확인

## 결론
100 고유 이미지의 한계는 증강으로 극복 불가.
딥러닝 외 접근 필요 → 고전 CV + OCR 탐색.
