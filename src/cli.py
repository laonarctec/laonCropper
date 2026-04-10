"""CLI — 영수증 크롭 명령행 인터페이스.

Usage:
    python -m src.cli input.jpg [output.jpg]
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.processor import ImageProcessor


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m src.cli <input_image> [output_image]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    if not Path(input_path).exists():
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    processor = ImageProcessor()
    result = processor.crop(input_path, output_path)
    print(f"Done: {result}")


if __name__ == "__main__":
    main()
