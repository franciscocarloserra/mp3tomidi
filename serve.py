"""MuScriptor service: bundled muscriptor FastAPI app + /convert. The panel (web/) is served apart.

POST /convert is /transcribe with a per-file cache in data/<sha1>.jsonl: a finished
file replays instantly, an interrupted one resumes from its last finished chunk.
"""
import asyncio
import base64
import hashlib
import io
import os
import json
import subprocess
import tempfile
import threading
from pathlib import Path

import uvicorn
import torch
from fastapi import File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from muscriptor.events import NoteEndEvent, NoteStartEvent, ProgressEvent
from muscriptor.server import create_app, event_to_dict
from muscriptor.transcription_model import _SEGMENT_DURATION, TranscriptionModel
from muscriptor.utils.audio import _read_non_wav_file

WEIGHTS = os.environ.get("MUSCRIPTOR_WEIGHTS", "large")  # "large" downloads from HF; or a local .safetensors path
PORT = 6977
HOST = "0.0.0.0"  # LAN, so /robot opens from a phone
ROBOT_HZ = 30  # robot stream rate
DATA = Path(__file__).parent / "data"
SEPARATE = Path(__file__).parent / "separate.py"  # htdemucs
DEMUCS_PYTHON = os.environ.get("DEMUCS_PYTHON", "python3")  # a python with demucs installed

DATA.mkdir(exist_ok=True)
model = TranscriptionModel.load_model(WEIGHTS)
app = create_app(model)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
lock = threading.Lock()
cancel = threading.Event()


def load_cache(path):
    """Cached lines up to the last finished chunk (later lines belong to a cut chunk)."""
    if not path.exists():
        return [], 0
    lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if lines and lines[-1]["type"] == "transcription_complete":
        return lines, None
    last = max((i for i, l in enumerate(lines) if l["type"] == "progress"), default=-1)
    done = lines[last]["completed"] if last >= 0 else 0
    return lines[: last + 1], done


@app.post("/convert")
async def convert(file: UploadFile = File()):
    data = await file.read()
    path = DATA / f"{hashlib.sha1(data).hexdigest()[:16]}.jsonl"
    cached, done = load_cache(path)
    if done is None:
        return StreamingResponse((f"data: {json.dumps(l)}\n\n" for l in cached), media_type="text/event-stream")
    try:
        wav, sr = _read_non_wav_file(io.BytesIO(data))
    except Exception:  # libsndfile can't read it (m4a, webm, opus…): let ffmpeg decode
        with tempfile.NamedTemporaryFile(suffix=Path(file.filename or "").suffix) as tmp:  # mp4 needs a seekable input
            tmp.write(data); tmp.flush()
            pcm = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", tmp.name, "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:"],
                                 capture_output=True).stdout
        if not pcm:
            raise HTTPException(400, f"no pude decodificar {file.filename}")
        wav, sr = torch.frombuffer(bytearray(pcm), dtype=torch.float32), 16000
    (DATA / f"{path.stem}.name").write_text(file.filename or "")

    def gen():
        cancel.set()
        with lock:
            cancel.clear()
            starts, events, open_ = {}, [], set()
            for l in cached:  # rebuild event objects from the cache
                if l["type"] == "start":
                    ev = NoteStartEvent(l["pitch"], l["start_time"], l["index"], l["instrument"])
                    starts[ev.index] = ev; events.append(ev); open_.add(ev.index)
                elif l["type"] == "end":
                    events.append(NoteEndEvent(l["end_time"], starts[l["start_event_index"]])); open_.discard(l["start_event_index"])
                yield f"data: {json.dumps(l)}\n\n"
            t0, idx0 = done * _SEGMENT_DURATION, max(starts, default=-1) + 1
            with path.open("w") as f:
                for l in cached:
                    f.write(json.dumps(l) + "\n")

                def emit(d):
                    f.write(json.dumps(d) + "\n"); f.flush()
                    return f"data: {json.dumps(d)}\n\n"

                for i in sorted(open_):  # notes left hanging at the cut: close them there
                    events.append(NoteEndEvent(t0, starts[i])); yield emit({"type": "end", "end_time": t0, "start_event_index": i})
                tail = wav[..., int(t0 * sr):] if wav.dim() > 1 else wav[int(t0 * sr):]
                mine = {}
                for ev in model.transcribe((tail, sr), batch_size=1, no_eos_is_ok=True):
                    if cancel.is_set():
                        return
                    if isinstance(ev, ProgressEvent):
                        yield emit({"type": "progress", "completed": ev.completed + done, "total": ev.total + done})
                        continue
                    if isinstance(ev, NoteStartEvent):
                        ev = NoteStartEvent(ev.pitch, ev.start_time + t0, ev.index + idx0, ev.instrument)
                        mine[ev.index] = ev
                    else:
                        ev = NoteEndEvent(ev.end_time + t0, mine[ev.start_event.index + idx0])
                    events.append(ev)
                    yield emit(event_to_dict(ev))
                grid = model.detect_beat_grid_for((wav, sr))
                if grid:
                    grid = grid.with_onset_delay([e.start_time for e in events if isinstance(e, NoteStartEvent)])
                midi = model.events_to_midi_bytes(iter(events), beat_grid=grid)
                (DATA / f"{path.stem}.mid").write_bytes(midi)
                yield emit({
                    "type": "transcription_complete",
                    "data": base64.b64encode(midi).decode("ascii"),
                    "beat_grid": {"bpm": grid.bpm, "onset_delay": grid.onset_delay} if grid else None,
                })

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.post("/voice")
def voice(file: UploadFile = File()):
    """Vocals stem via htdemucs (subprocess, so its VRAM is freed after), cached in data/<sha1>_stems/."""
    data = file.file.read()
    out = DATA / f"{hashlib.sha1(data).hexdigest()[:16]}_stems"
    if not (out / "vocals.opus").exists():
        with tempfile.NamedTemporaryFile(suffix=Path(file.filename or "").suffix) as tmp:
            tmp.write(data); tmp.flush()
            r = subprocess.run([DEMUCS_PYTHON, str(SEPARATE), tmp.name, str(out)], capture_output=True, text=True)
        if r.returncode:
            raise HTTPException(500, r.stderr[-500:])
    return FileResponse(out / "vocals.opus", media_type="audio/ogg")

robot = {"level": 0.0}  # vocals loudness relayed from the panel to /robot viewers


@app.post("/robot/level")
async def robot_level(level: float):
    robot["level"] = level


@app.get("/robot/stream")
async def robot_stream():
    async def gen():
        while True:
            yield f"data: {robot['level']:.3f}\n\n"
            await asyncio.sleep(1 / ROBOT_HZ)
    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/robot")
def robot_page():
    return HTMLResponse((Path(__file__).parent / "web" / "robot.html").read_text())

uvicorn.run(app, host=HOST, port=PORT)
