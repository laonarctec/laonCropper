# 08. 최종 의사결정

## 선택: OCR DensTuned + 하이브리드 fallback

### 파이프라인
```
OCR DensTuned (1차)
    ↓ 실패시
ContourDetector (2차)
    ↓ 실패시
Cliff Scan (3차)
```

### 왜 이 조합인가
1. **OCR DensTuned**: 학습 불필요, 일반화 우수, 6/6 중 5/6 완벽
2. **ContourDetector**: OCR 실패시 (텍스트 없는 이미지 등) 안전망
3. **Cliff Scan**: 최후 fallback, 기존 로직 보존

### 폐기된 접근
| 접근 | 폐기 사유 |
|------|-----------|
| YOLO 원본 (100장) | 학습 데이터 편향 (bbox 면적 86%) → 실샘플 전혀 일반화 안됨 |
| 증강 v1 (500장) | 100 고유 이미지의 콘텐츠 다양성 한계 → 과적합 (Box Loss gap 0.33) |
| 라인강조 + 증강 v2 (1000장) | 동일 과적합, Overfitting Monitor ALERT |
| OCR Basic | 배경 텍스트(포스터/간판) 포함 문제 |
| OCR Heatmap | 너무 공격적 → 영수증 상하단 잘림 |
| OCR Density (튜닝 전) | 세로 긴 영수증 잘림 (eps 고정 한계) |

### 최종 구조 (SoC)
```
final/
├── detection/          # 탐지 레이어
│   ├── bbox.py         # BBox 타입
│   ├── ocr.py          # OCR DensTuned (primary)
│   ├── contour.py      # 컨투어 (fallback #1)
│   └── cliff.py        # Cliff Scan (fallback #2)
├── processor.py        # 크롭 오케스트레이터
├── server.py           # FastAPI
├── cli.py              # CLI
└── README.md
```

## 교훈
1. 소규모 데이터셋(100장)으로 딥러닝 객체 탐지 일반화는 구조적 한계
2. 증강은 콘텐츠 다양성을 대체할 수 없음
3. "영수증 = 텍스트 집합"이라는 도메인 지식이 범용 모델(YOLO)보다 효과적
4. 사전학습된 텍스트 탐지기(CRAFT/EasyOCR)가 100장 fine-tune보다 강건
