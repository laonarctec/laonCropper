# laonCropper

영수증 이미지에서 배경을 제거하고 영수증 영역만 자동 크롭하는 API 서버.

OCR 텍스트 밀도 클러스터링(DensTuned) 기반으로 학습 데이터 없이 동작합니다.

## 실행

### Docker (권장)
```bash
docker compose up -d
# http://localhost:8200/docs 에서 Swagger UI로 테스트
```

### 로컬
```bash
uv sync
uv run python -m src.server
```

### CLI
```bash
uv run python -m src.cli input.jpg output.jpg
```

## API

### `GET /health`

서버 상태 확인.

```bash
curl http://localhost:8200/health
# {"status":"ok"}
```

### `POST /crop`

영수증 이미지를 업로드하면 크롭된 이미지를 반환합니다.

| 항목 | 값 |
|------|---|
| Method | `POST` |
| URL | `http://localhost:8200/crop` |
| Content-Type | `multipart/form-data` |
| 파라미터 | `file` (이미지 파일) |
| 응답 | `image/jpeg` (크롭된 이미지 바이너리) |
| 상태코드 | 200 성공 / 422 파일 누락 |

#### 호출 예시

```bash
# cURL
curl -X POST http://localhost:8200/crop \
  -F "file=@영수증.jpg" \
  -o 결과.jpg
```

```javascript
// JavaScript (fetch)
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await fetch('http://서버주소:8200/crop', {
  method: 'POST',
  body: formData,
});

const blob = await response.blob();
const url = URL.createObjectURL(blob);
```

```python
# Python (requests)
import requests

with open("영수증.jpg", "rb") as f:
    r = requests.post("http://서버주소:8200/crop", files={"file": f})

with open("결과.jpg", "wb") as f:
    f.write(r.content)
```

## 탐지 파이프라인

3단 fallback으로 안정성 확보:

```
[1차] OCR DensTuned
  │ EasyOCR 텍스트 검출 (960px 리사이즈)
  │ → 큰 텍스트 필터 (면적 > 중앙값 × 10 제거)
  │ → y가중(0.5) DBSCAN 클러스터링
  │ → y범위 겹치는 인접 클러스터 병합
  │ → 최대 클러스터 bbox + 2% padding
  ↓ 실패시
[2차] ContourDetector
  │ Canny + adaptiveThreshold → morphology → 최대 사각형
  ↓ 실패시
[3차] CliffScanner
  │ Sobel 에지 → 중앙→외곽 에너지 절벽 탐색
```

### DensTuned 핵심 알고리즘

1. **텍스트 검출**: EasyOCR CRAFT 모델 (수백만 장으로 사전학습, 추가 학습 불필요)
2. **크기 필터**: 면적 > 중앙값 × 10인 bbox 제거 (포스터/간판 큰 글자 배제)
3. **y가중 DBSCAN**: y좌표를 0.5로 축소해 세로 긴 영수증의 줄 간 연결 촉진 (eps = scaled 대각선 × 10%)
4. **클러스터 병합**: 최대 클러스터와 y범위 30% 이상 겹치는 인접 클러스터 병합 (좌/우 분리 문제 해결)
5. **bbox 추출**: 병합된 클러스터의 최외곽 포인트 + 2% padding

## 프로젝트 구조

```
laonCropper/
├── src/                        # 소스 코드 (SoC)
│   ├── detection/              #   탐지 레이어
│   │   ├── bbox.py             #     BBox 공유 타입
│   │   ├── ocr.py              #     OCR DensTuned (primary)
│   │   ├── contour.py          #     컨투어 (fallback #1)
│   │   └── cliff.py            #     Cliff Scan (fallback #2)
│   ├── processor.py            #   크롭 오케스트레이터 (3단 fallback)
│   ├── server.py               #   FastAPI (GET /health, POST /crop)
│   └── cli.py                  #   CLI
├── docs/
│   ├── worklog/                #   개발 과정 기록 (8단계)
│   └── archive/                #   PDCA 아카이브
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
└── pyproject.toml
```

### SoC (Separation of Concerns)

| 레이어 | 파일 | 책임 |
|--------|------|------|
| 탐지 | `detection/*.py` | BBox 반환만. 크롭/I/O 책임 없음 |
| 처리 | `processor.py` | 탐지 순서 결정 + 크롭 + 파일 저장 |
| API | `server.py` | HTTP 입출력만. 탐지/크롭 로직 모름 |
| CLI | `cli.py` | 명령행 진입점. 파일 경로만 전달 |

## Docker 운영

```bash
# 시작
docker compose up -d

# 중지
docker compose down

# 로그 확인
docker compose logs -f

# 재빌드 (코드 수정 후)
docker compose up -d --build

# 헬스체크 상태
docker compose ps
```

## 개발 과정

YOLO fine-tuning → 과적합 → 증강 실험 → OCR 텍스트 밀도 클러스터링으로 전환.
자세한 내용은 [`docs/worklog/`](docs/worklog/) 참조.

| 단계 | 접근 | 결과 |
|------|------|------|
| 01 | Plan & Design | YOLO + fallback 아키텍처 설계 |
| 02 | YOLO 학습 (100장) | mAP 0.97이지만 실샘플 실패 (데이터 편향) |
| 03 | Synthetic Padding 증강 | 과적합 (Box Loss gap 0.33) |
| 04 | 라인 강조 + 증강 v2 | 여전히 과적합 |
| 05 | 컨투어 검출 | 개선되었으나 불충분 |
| 06 | OCR 텍스트 검출 | 4/6 완벽, 2/6 배경 텍스트 포함 |
| 07 | DBSCAN 밀도 클러스터링 | DensTuned 방법 확정 |
| 08 | 최종 결정 + Docker 배포 | 3단 fallback + 클러스터 병합 |

### 핵심 교훈

- 소규모 데이터셋(100장)으로 딥러닝 일반화는 구조적 한계
- 증강은 콘텐츠 다양성을 대체할 수 없음
- "영수증 = 텍스트 집합"이라는 도메인 지식이 범용 모델(YOLO)보다 효과적
- 사전학습된 텍스트 탐지기(CRAFT/EasyOCR)가 100장 fine-tune보다 강건

## 기술 스택

| 기술 | 용도 |
|------|------|
| **EasyOCR** | CRAFT 기반 텍스트 검출 (사전학습) |
| **scikit-learn** | DBSCAN 클러스터링 |
| **OpenCV** | 이미지 처리, 컨투어 검출 |
| **FastAPI** | API 서버 |
| **Docker** | 컨테이너 배포 |
| **uv** | Python 패키지 관리 |

## 라이선스

Private — laonarctec
