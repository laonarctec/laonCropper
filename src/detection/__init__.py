"""Detection layer — 영수증 영역 탐지 모듈.

3가지 탐지기를 제공하며, processor가 우선순위대로 시도한다:
    1. OcrDensityDetector  — OCR 텍스트 bbox 기반 (primary)
    2. ContourDetector     — 윤곽선 기반 (fallback #1)
    3. CliffScanner        — 에지 에너지 기반 (fallback #2)
"""

from src.detection.bbox import BBox
from src.detection.cliff import CliffScanner
from src.detection.contour import ContourDetector
from src.detection.ocr import OcrDensityDetector

__all__ = ["BBox", "OcrDensityDetector", "ContourDetector", "CliffScanner"]
