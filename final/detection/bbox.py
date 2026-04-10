"""탐지 결과 공유 타입."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BBox:
    """탐지된 영수증 bounding box (원본 이미지 좌표계, 픽셀 단위)."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    method: str = ""  # 어떤 탐지기가 생성했는지 ("ocr", "contour", "cliff")
