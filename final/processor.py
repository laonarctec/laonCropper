"""영수증 크롭 오케스트레이터.

3단 fallback 파이프라인: OCR DensTuned → Contour → Cliff Scan.
각 탐지 레이어는 BBox를 반환하고, 이 모듈이 크롭 + 파일 저장을 담당한다.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from final.detection import BBox, CliffScanner, ContourDetector, OcrDensityDetector

logger = logging.getLogger(__name__)


class ImageProcessor:
    """3단 fallback 영수증 크롭 프로세서.

    Usage:
        processor = ImageProcessor()
        output_path = processor.crop("input.jpg", "output.jpg")
    """

    def __init__(
        self,
        ocr: OcrDensityDetector | None = None,
        contour: ContourDetector | None = None,
        cliff: CliffScanner | None = None,
        pad_ratio: float = 0.02,
    ) -> None:
        self._ocr = ocr or OcrDensityDetector()
        self._contour = contour or ContourDetector()
        self._cliff = cliff or CliffScanner()
        self._pad_ratio = pad_ratio

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def crop(
        self, image_path: str, output_path: str | None = None
    ) -> str:
        """이미지에서 영수증 영역을 크롭하여 저장. 저장 경로 반환."""
        img = cv2.imread(image_path)
        if img is None:
            return image_path

        bbox = self._detect(img)
        result = self._apply_crop(img, bbox)

        out = self._resolve_output_path(image_path, output_path, bbox.method)
        if not cv2.imwrite(out, result):
            raise IOError(f"cv2.imwrite failed: {out}")
        return out

    # ------------------------------------------------------------------ #
    # Detection — 3단 fallback
    # ------------------------------------------------------------------ #

    def _detect(self, img: np.ndarray) -> BBox:
        """OCR → Contour → Cliff 순서로 시도. 항상 BBox 반환."""

        # 1차: OCR DensTuned
        bbox = self._safe_call(self._ocr, img)
        if bbox is not None:
            logger.info(f"[1차 OCR] 성공: {bbox.method}")
            return bbox

        # 2차: Contour
        bbox = self._safe_call(self._contour, img)
        if bbox is not None:
            logger.info(f"[2차 Contour] OCR 실패 → Contour 성공")
            return bbox

        # 3차: Cliff (항상 반환)
        bbox = self._safe_call(self._cliff, img)
        if bbox is not None:
            logger.info(f"[3차 Cliff] OCR+Contour 실패 → Cliff 사용")
            return bbox

        # Cliff도 실패시 (이론상 없음) → 전체 이미지
        h, w = img.shape[:2]
        return BBox(x1=0, y1=0, x2=w, y2=h, confidence=0.0, method="none")

    def _safe_call(self, detector, img: np.ndarray) -> BBox | None:
        try:
            return detector.detect(img)
        except Exception as e:
            logger.warning(f"{detector.__class__.__name__} error: {e}")
            return None

    # ------------------------------------------------------------------ #
    # Crop & Output
    # ------------------------------------------------------------------ #

    def _apply_crop(self, img: np.ndarray, bbox: BBox) -> np.ndarray:
        """BBox 영역 크롭. padding 적용."""
        h, w = img.shape[:2]
        pad_x = int(w * self._pad_ratio)
        pad_y = int(h * self._pad_ratio)
        x1 = max(0, bbox.x1 - pad_x)
        y1 = max(0, bbox.y1 - pad_y)
        x2 = min(w, bbox.x2 + pad_x)
        y2 = min(h, bbox.y2 + pad_y)
        return img[y1:y2, x1:x2]

    def _resolve_output_path(
        self, image_path: str, output_path: str | None, method: str
    ) -> str:
        if output_path is not None:
            return output_path
        p = Path(image_path)
        return str(p.parent / f"cropped_{p.name}")
