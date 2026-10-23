import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

import cv2
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from segment import Cancelled, Segmenter, first_frame

ROOT = Path(__file__).parent
JOBS = Path(tempfile.gettempdir()) / "vbr-jobs"
MAX_UPLOAD = 500 * 1024 * 1024
MAX_AGE = 2 * 3600
MAX_SECONDS = float(
    os.environ.get("MAX_SECONDS", 0)
)  # 0 = no limit; set on shared servers

segmenter = Segmenter()
work_lock = threading.Lock()  # one video at a time; others wait in line
jobs = {}

app = FastAPI()


class SegmentRequest(BaseModel):
    box: list[int] = Field(min_length=4, max_length=4)
    color: str = Field(pattern=r"^#[0-9a-fA-F]{6}$")
    fast: bool = False


def job_dir(job_id):
    if not re.fullmatch(r"[0-9a-f]{32}", job_id) or not (JOBS / job_id).is_dir():
        raise HTTPException(404, "Unknown video")
    return JOBS / job_id


def clean_old_jobs():
    for d in JOBS.glob("*"):
        if time.time() - d.stat().st_mtime > MAX_AGE:
            shutil.rmtree(d, ignore_errors=True)
            jobs.pop(d.name, None)


def prepare(d):
    try:
        frame = first_frame(str(d / "input"))
    except ValueError:
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(400, "Can't read this file as a video")
    cap = cv2.VideoCapture(str(d / "input"))
    seconds = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 25)
    cap.release()
    if MAX_SECONDS and seconds > MAX_SECONDS:
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(
            413,
            f"This demo takes videos up to {MAX_SECONDS:g} seconds. "
            "Run it on your own computer for longer ones.",
        )
    cv2.imwrite(
        str(d / "frame.jpg"),
        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )
    height, width = frame.shape[:2]
    return {
        "id": d.name,
        "width": width,
        "height": height,
        "subjects": segmenter.detect(frame)[:10],
    }


def new_job():
    clean_old_jobs()
    d = JOBS / uuid.uuid4().hex
    d.mkdir(parents=True)
    return d


@app.post("/api/videos")
def upload(file: UploadFile):
    d = new_job()
    size = 0
    with open(d / "input", "wb") as f:
        while chunk := file.file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD:
                shutil.rmtree(d, ignore_errors=True)
                raise HTTPException(413, "Videos up to 500 MB only")
            f.write(chunk)
    return prepare(d)


@app.post("/api/example")
def example():
    d = new_job()
    shutil.copy(ROOT / "assets" / "example.mp4", d / "input")
    return prepare(d)


@app.get("/api/videos/{job_id}/frame.jpg")
def frame(job_id: str):
    return FileResponse(job_dir(job_id) / "frame.jpg")


@app.post("/api/videos/{job_id}/segment", status_code=202)
def start(job_id: str, req: SegmentRequest):
    d = job_dir(job_id)
    if jobs.get(job_id, {}).get("status") in ("waiting", "running"):
        raise HTTPException(409, "Already running")
    x0, y0, x1, y1 = req.box
    if x1 - x0 < 4 or y1 - y0 < 4:
        raise HTTPException(422, "Box too small")
    cancel = threading.Event()
    state = jobs[job_id] = {"status": "waiting", "done": 0, "total": 0, "fps": 0.0}
    state["cancel"] = cancel

    def progress(done, total):
        state.update(
            done=done, total=total, fps=round(done / (time.time() - state["start"]), 1)
        )

    def run():
        with work_lock:
            if cancel.is_set():
                return
            state.update(status="running", start=time.time())
            try:
                segmenter.segment(
                    str(d / "input"),
                    str(d / "result.mp4"),
                    req.box,
                    req.color,
                    req.fast,
                    progress,
                    cancel,
                )
                state["status"] = "done"
            except Cancelled:
                pass
            except Exception as e:
                state.update(status="error", error=str(e))

    threading.Thread(target=run, daemon=True).start()
    return {"status": "waiting"}


@app.post("/api/videos/{job_id}/cancel")
def cancel(job_id: str):
    job_dir(job_id)
    state = jobs.get(job_id)
    if state and state["status"] in ("waiting", "running"):
        state["cancel"].set()
        state["status"] = "cancelled"
    return {"status": "cancelled"}


@app.get("/api/videos/{job_id}/status")
def status(job_id: str):
    job_dir(job_id)
    state = jobs.get(job_id, {"status": "idle"})
    return {k: v for k, v in state.items() if k not in ("start", "cancel")}


@app.get("/api/videos/{job_id}/result.mp4")
def result(job_id: str):
    path = job_dir(job_id) / "result.mp4"
    if jobs.get(job_id, {}).get("status") != "done":
        raise HTTPException(404, "Not ready")
    return FileResponse(path, media_type="video/mp4", filename="background-removed.mp4")


app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
