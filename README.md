# mp3tomidi

Drop a song, get MIDI. A small web panel on top of [MuScriptor](https://github.com/muscriptor/muscriptor) (audio → multi-instrument MIDI) and [Demucs](https://github.com/facebookresearch/demucs) htdemucs (vocal separation).

Demo (static, no model running): a cached transcription plus its separated vocals, at the repo's GitHub Pages URL.

## Pipeline

1. The panel uploads the file to `serve.py` (`POST /convert`).
2. MuScriptor transcribes it in 5 s chunks and streams note on/off events back over SSE, so the piano roll fills in while it runs. Each instrument it detects gets its own track. At the end it detects the beat grid (beat_this) and builds the .mid.
3. "split voz" sends the same file to `POST /voice`, which runs `separate.py` (htdemucs, GPU) and returns the vocals stem. The panel plays it on top of the MIDI, aligned using the onset delay from the beat grid.
4. Playback is a basic WebAudio synth, or Web MIDI out to a real keyboard (one channel and GM program per instrument, drums on channel 10).

Results are cached per file in `data/<sha1>.jsonl` (+ `.mid`, `_stems/`). A finished file replays instantly; an interrupted one resumes from its last finished chunk.

## Run

    python -m venv .venv && .venv/bin/pip install muscriptor
    pip install demucs          # any python; point DEMUCS_PYTHON at it (default: python3)
    .venv/bin/python serve.py   # service on :6977
    cd web && python3 -m http.server 7808   # panel on http://localhost:7808

`MUSCRIPTOR_WEIGHTS` picks the model: `large` (default, downloaded from HuggingFace, needs accepting its license there) or a local `.safetensors` path. Large takes ~11 GB of VRAM; on an RTX 3090, 15 s of audio takes ~25 s.

Panel: click the roll to play/stop, ctrl+wheel or +/- to zoom. On a track: click mutes, double click solos, right click puts it on MIDI channel 1.

`/robot` (from a phone on the same network: `http://<this-ip>:6977/robot`) is a small 3D robot that dances to the vocals while the panel plays them.

## Limitations

MuScriptor gives notes per instrument but no velocity. The synth is a toy, not a soundfont. The demo page can't transcribe new files.
