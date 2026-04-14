"""OCR DensTuned 영수증 검출기.

파이프라인:
    1. EasyOCR text detection (960px 리사이즈)
    2. 큰 텍스트 bbox 필터 (면적 > 중앙값 × 5 제거)
    3. y좌표 0.5 스케일 → DBSCAN (eps = scaled_diag × 10%)
    4. 최대 클러스터의 bounding rect + 2% padding
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

from src.detection.bbox import BBox

logger = logging.getLogger(__name__)

MIN_TEXT_BOXES = 3
PAD_RATIO = 0.02
DET_MAX_SIZE = 960
AREA_FILTER_MULT = 10.0
Y_SCALE = 0.5
EPS_RATIO = 0.10
MIN_SAMPLES = 3


class OcrDensityDetector:
    """EasyOCR + DBSCAN Density 클러스터링 기반 영수증 탐지."""

    def __init__(self) -> None:
        self._reader = None

    def _ensure_loaded(self) -> None:
        if self._reader is None:
            import easyocr

            self._reader = easyocr.Reader(
                ["ko", "en"], gpu=False, verbose=False
            )
            logger.info("EasyOCR text detector loaded")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def detect(self, image: np.ndarray) -> BBox | None:
        """이미지에서 텍스트 밀집 영역(영수증)을 찾아 BBox 반환."""
        self._ensure_loaded()
        assert self._reader is not None

        h, w = image.shape[:2]
        boxes = self._detect_text_boxes(image, h, w)

        if len(boxes) < MIN_TEXT_BOXES:
            return None

        bbox = self._density_cluster(boxes, h, w)
        if bbox is not None:
            logger.info(
                f"OCR DensTuned: {len(boxes)}개 텍스트 → "
                f"bbox=({bbox.x1},{bbox.y1},{bbox.x2},{bbox.y2})"
            )
        return bbox

    # ------------------------------------------------------------------ #
    # Private
    # ------------------------------------------------------------------ #

    def _detect_text_boxes(
        self, image: np.ndarray, h: int, w: int
    ) -> list[tuple[int, int, int, int]]:
        """EasyOCR detect → (x1, y1, x2, y2) 리스트. 리사이즈 역산 포함."""
        scale = 1.0
        det_img = image
        if max(h, w) > DET_MAX_SIZE:
            scale = DET_MAX_SIZE / max(h, w)
            det_img = cv2.resize(
                image,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )

        h_list, f_list = self._reader.detect(det_img)

        boxes: list[tuple[int, int, int, int]] = []
        if h_list and h_list[0]:
            for x_min, x_max, y_min, y_max in h_list[0]:
                boxes.append((
                    int(x_min / scale),
                    int(y_min / scale),
                    int(x_max / scale),
                    int(y_max / scale),
                ))
        if f_list and f_list[0]:
            for poly in f_list[0]:
                pts = np.array(poly)
                boxes.append((
                    int(pts[:, 0].min() / scale),
                    int(pts[:, 1].min() / scale),
                    int(pts[:, 0].max() / scale),
                    int(pts[:, 1].max() / scale),
                ))
        return boxes

    def _density_cluster(
        self,
        boxes: list[tuple[int, int, int, int]],
        h: int,
        w: int,
    ) -> BBox | None:
        """크기 필터 + y가중 DBSCAN → 최대 클러스터 bbox."""
        # 1. 크기 필터: 큰 텍스트 (포스터/간판) 제거
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        median_area = float(np.median(areas))
        threshold = median_area * AREA_FILTER_MULT
        filtered = [b for b, a in zip(boxes, areas) if a < threshold]

        if len(filtered) < MIN_TEXT_BOXES:
            return None

        # 2. y가중 스케일링 (세로 긴 영수증의 줄 간 연결 촉진)
        centers = np.array([
            ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in filtered
        ])
        scaled = centers.copy()
        scaled[:, 1] *= Y_SCALE

        # 3. DBSCAN 클러스터링
        diag = np.sqrt(w**2 + (h * Y_SCALE) ** 2)
        eps = diag * EPS_RATIO

        db = DBSCAN(eps=eps, min_samples=MIN_SAMPLES).fit(scaled)
        labels = db.labels_

        unique_labels = set(labels)
        unique_labels.discard(-1)
        if not unique_labels:
            return None

        best_label = max(unique_labels, key=lambda l: np.sum(labels == l))
        best_mask = labels == best_label
        best_boxes = [b for b, m in zip(filtered, best_mask) if m]

        if len(best_boxes) < MIN_TEXT_BOXES:
            return None

        # 4. y 범위 겹치거나 인접한 클러스터 병합
        #    영수증은 좌(항목명) / 우(금액)로 분리될 수 있으나 y 범위는 동일.
        #    상단(품목) / 하단(합계)처럼 수직으로 인접한 경우도 병합.
        #    포스터/간판 텍스트는 y 간격이 크므로 병합되지 않음.
        best_ys = [b[1] for b in best_boxes] + [b[3] for b in best_boxes]
        best_y_min, best_y_max = min(best_ys), max(best_ys)
        best_y_span = best_y_max - best_y_min

        avg_line_h = float(np.mean([b[3] - b[1] for b in filtered]))
        gap_threshold = avg_line_h * 2.5

        cluster_boxes = list(best_boxes)
        for other_label in unique_labels:
            if other_label == best_label:
                continue
            other_mask = labels == other_label
            other_boxes = [b for b, m in zip(filtered, other_mask) if m]
            other_ys = [b[1] for b in other_boxes] + [b[3] for b in other_boxes]
            other_y_min, other_y_max = min(other_ys), max(other_ys)
            overlap = max(0, min(best_y_max, other_y_max) - max(best_y_min, other_y_min))
            overlap_ratio = overlap / best_y_span if best_y_span > 0 else 0
            gap = max(0, max(best_y_min, other_y_min) - min(best_y_max, other_y_max))
            if overlap_ratio > 0.3 or gap < gap_threshold:
                cluster_boxes.extend(other_boxes)

        # 5. 최외곽 포인트 + padding
        xs = [b[0] for b in cluster_boxes] + [b[2] for b in cluster_boxes]
        ys = [b[1] for b in cluster_boxes] + [b[3] for b in cluster_boxes]
        pad_x = int(w * PAD_RATIO)
        pad_y = int(h * PAD_RATIO)

        return BBox(
            x1=max(0, min(xs) - pad_x),
            y1=max(0, min(ys) - pad_y),
            x2=min(w, max(xs) + pad_x),
            y2=min(h, max(ys) + pad_y),
            confidence=min(1.0, len(cluster_boxes) / 10.0),
            method="ocr",
        )
