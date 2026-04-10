# 05. 컨투어 기반 검출

## 접근
학습 없이 동작하는 고전 CV:
Canny edge → adaptiveThreshold → morphology → findContours → 최대 사각형.

## 구현
- `contour_detector.py`: ContourDetector 클래스
- 면적 필터 (MIN 3% ~ MAX 92%)
- 사각형 유사도 + 4각형 보너스 스코어링

## 실샘플 결과
- 6/6 탐지 성공
- YOLO보다 개선되었으나, 일부 이미지에서 배경 포함
- 속도: 317~432ms (Cliff보다 느림)

## 사용자 평가
"paddleOCR을 이용하여 텍스트 BBox 좌표를 기점으로 최외곽 포인트를 얻는 방식이 좋겠다"
→ OCR 기반 접근으로 전환
