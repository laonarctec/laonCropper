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

### `POST /crop`

영수증 이미지를 업로드하면 크롭된 이미지를 반환합니다.

| 항목 | 값 |
|------|---|
| Content-Type | `multipart/form-data` |
| 파라미터 | `file` (이미지 파일) |
| 응답 | `image/jpeg` (크롭된 이미지) |

```bash
curl -X POST http://localhost:8200/crop \
  -F "file=@영수증.jpg" \
  -o 결과.jpg
```

## 탐지 파이프라인

3단 fallback으로 안정성 확보:

```
[1차] OCR DensTuned ── 텍스트 bbox 클러스터링 → 영수증 영역
  ↓ 실패시
[2차] ContourDetector ── 윤곽선 기반 최대 사각형
  ↓ 실패시
[3차] CliffScanner ── 에지 에너지 절벽 탐색
```

## 프로젝트 구조

```
├── src/                    # 소스 코드 (SoC)
│   ├── detection/          #   탐지 레이어
│   │   ├── ocr.py          #     OCR DensTuned (primary)
│   │   ├── contour.py      #     컨투어 (fallback #1)
│   │   └── cliff.py        #     Cliff Scan (fallback #2)
│   ├── processor.py        #   크롭 오케스트레이터
│   ├── server.py           #   FastAPI
│   └── cli.py              #   CLI
├── docs/
│   ├── worklog/            #   개발 과정 기록 (8단계)
│   └── archive/            #   PDCA 아카이브
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `OCR_PAD_RATIO` | `0.02` | 텍스트 영역 외곽 padding 비율 |

## 기술 스택

- **EasyOCR** — CRAFT 기반 텍스트 검출 (사전학습, 학습 불필요)
- **scikit-learn** — DBSCAN 클러스터링
- **OpenCV** — 이미지 처리, 컨투어 검출
- **FastAPI** — API 서버
- **Docker** — 컨테이너 배포
