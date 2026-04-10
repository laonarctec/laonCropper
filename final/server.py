"""FastAPI 서버 — 영수증 크롭 API.

엔드포인트:
    POST /crop — 이미지 업로드 → 크롭된 이미지 반환
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from final.processor import ImageProcessor

app = FastAPI(title="laonCropper API")
_processor = ImageProcessor()

UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.post("/crop")
async def crop_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    ext = Path(file.filename or "").suffix
    tmp_name = f"{uuid.uuid4()}{ext}"
    tmp_path = UPLOAD_DIR / tmp_name

    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    output_path = UPLOAD_DIR / f"cropped_{tmp_name}"
    final_path = await run_in_threadpool(
        _processor.crop, str(tmp_path), str(output_path)
    )

    background_tasks.add_task(tmp_path.unlink, missing_ok=True)
    background_tasks.add_task(Path(final_path).unlink, missing_ok=True)

    return FileResponse(
        final_path,
        media_type="image/jpeg",
        filename=f"crop_{file.filename}",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8200)
