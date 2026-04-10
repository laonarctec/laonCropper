"""컨투어 기반 영수증 검출기.

Canny + adaptiveThreshold → morphology → findContours → 최대 사각형.
학습 데이터 불필요. OCR 실패시 fallback으로 사용.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from final.detection.bbox import BBox

logger = logging.getLogger(__name__)

MIN_AREA_RATIO = 0.03
MAX_AREA_RATIO = 0.92


class ContourDetector:
    """윤곽선 기반 영수증 사각형 검출."""

    def detect(self, image: np.ndarray) -> BBox | None:
        h, w = image.shape[:2]
        total_area = h * w

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 8,
        )
        edges_canny = cv2.Canny(blurred, 30, 120)
        edges = cv2.bitwise_or(thresh, edges_canny)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        edges = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        best_bbox: BBox | None = None
        best_score: float = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            ratio = area / total_area
            if ratio < MIN_AREA_RATIO or ratio > MAX_AREA_RATIO:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
            x, y, bw, bh = cv2.boundingRect(cnt)
            rect_area = bw * bh
            rect_fill = area / rect_area if rect_area > 0 else 0
            quad_bonus = 1.3 if len(approx) == 4 else 1.0
            score = area * rect_fill * quad_bonus

            if score > best_score:
                best_score = score
                best_bbox = BBox(
                    x1=x, y1=y, x2=x + bw, y2=y + bh,
                    confidence=min(1.0, rect_fill * quad_bonus),
                    method="contour",
                )

        if best_bbox is not None:
            logger.info(
                f"Contour: conf={best_bbox.confidence:.2f} "
                f"bbox=({best_bbox.x1},{best_bbox.y1},{best_bbox.x2},{best_bbox.y2})"
            )
        return best_bbox
