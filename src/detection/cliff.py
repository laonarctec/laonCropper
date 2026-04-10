"""Center-Outward Cliff Scan 기반 영수증 크롭.

중앙에서 밖으로 나가며 에너지가 급감하는 "절벽"을 찾아 경계로 삼는다.
최후 fallback으로만 사용. 다른 탐지기와 달리 BBox가 아닌 크롭 이미지를 직접 반환.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.detection.bbox import BBox

logger = logging.getLogger(__name__)


class CliffScanner:
    """에너지 절벽 스캔으로 영수증 영역 추정."""

    def detect(self, image: np.ndarray) -> BBox | None:
        """에지 에너지 프로파일 분석으로 BBox 반환."""
        h, w = image.shape[:2]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sx = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
        sy = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
        freq_map = cv2.addWeighted(sx, 0.5, sy, 0.5, 0)

        cx, cy = w // 2, h // 2
        v_proj = np.mean(freq_map[cy - h // 5 : cy + h // 10, :], axis=0)
        h_proj = np.mean(freq_map[:, cx - w // 5 : cx + w // 5], axis=1)

        def find_cliff(profile, start, step, limit):
            body_e = np.mean(
                profile[max(0, start - 100) : min(len(profile), start + 100)]
            )
            threshold = body_e * 0.20
            look_ahead = 150
            for i in range(start, limit, step):
                win = profile[
                    min(i, i + step * look_ahead) : max(
                        i, i + step * look_ahead
                    ) : abs(step)
                ]
                if len(win) > 0 and np.mean(win) < threshold:
                    return i
            return limit

        l = find_cliff(v_proj, cx, -1, 0)
        r = find_cliff(v_proj, cx, 1, w - 1)
        t = find_cliff(h_proj, cy, -1, 0)
        b = find_cliff(h_proj, cy, 1, h - 1)

        pad_l = int(w * 0.02)
        pad_r = int(w * 0.06)
        pad_y = int(h * 0.04)

        x1 = max(0, l - pad_l)
        x2 = min(w, r + pad_r)
        y1 = max(0, t - pad_y)
        y2 = min(h, b + pad_y)
        x2 = min(x2, int(w * 0.85))

        logger.info(f"Cliff: bbox=({x1},{y1},{x2},{y2})")
        return BBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=0.5, method="cliff")
