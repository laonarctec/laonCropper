# 06. OCR 기반 검출

## 핵심 아이디어 (사용자 제안)
"영수증 = 텍스트의 집합. 텍스트 bbox의 최외곽 = 영수증 경계."
- OCR text detector는 수백만 장으로 학습됨 → 100장 과적합 문제 없음
- 학습 불필요, 배경/조명 무관

## 구현
- PaddleOCR 3.4.0: CPU 호환성 에러 (NotImplementedError) → **EasyOCR로 전환**
- `ocr_detector.py`: EasyOCR Reader, detect() (detection only), 960px 리사이즈
- 30초 → **4~5초**로 속도 최적화 (이미지 리사이즈 + detection-only 모드)

## 기본(Basic) 방법 결과
- 4/6 완벽 크롭 (2, 4, 5, 6)
- 2/6 넓은 크롭 (1: 포스터 텍스트 포함, 3: 키보드 텍스트 포함)
