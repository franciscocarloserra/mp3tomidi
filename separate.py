#!/usr/bin/env python3
"""GPU stem separation via Demucs.

Usage:
    python separate.py input.mp3 [output_dir] [model]

Output: {output_dir}/{stem}.opus  (never wav — see AUDIO_* below)
"""
import sys
import os
import time
import json
import contextlib
import subprocess
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
import soundfile as sf

# --- Tunable parameters ---
MODEL = "htdemucs"          # htdemucs (4 stems) | htdemucs_6s (+guitar, piano)
OVERLAP = 0.1               # demucs segment overlap (default .25); .1 is ~23%
                            # faster and indistinguishable on listening
FP16 = True                 # autocast; ~17% faster, indistinguishable
AUDIO_CODEC = "libopus"     # derivatives are never left as wav
AUDIO_BITRATE = "96k"
AUDIO_EXT = "opus"
AUDIO_EFFORT = "0"          # libopus -compression_level: encoder search effort,
                            # not quality at a given bitrate. 0 encodes the six
                            # stems in 3.7s against 6.2s at the default 10, for
                            # the same 22.5 MB. ffmpeg's native aac was 14s.
# --------------------------


def separate(input_path, output_dir=None, model_name=MODEL, overlap=OVERLAP,
             fp16=FP16):
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    if output_dir is None:
        output_dir = os.path.splitext(input_path)[0] + "_stems"
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    t0 = time.perf_counter()
    model = get_model(model_name)
    model.to(device)
    model_time = time.perf_counter() - t0
    print(f"Model loaded: {model_time:.1f}s")

    # Decode with ffmpeg straight to the model's rate. soundfile cannot read
    # m4a at all, and librosa falls back to the deprecated audioread path,
    # which measured 3.4s vs 0.7s here for a 5.6 min track.
    t0 = time.perf_counter()
    target_sr = model.samplerate
    raw = subprocess.run(
        ["ffmpeg", "-i", input_path, "-f", "f32le", "-acodec", "pcm_f32le",
         "-ar", str(target_sr), "-ac", "2", "-loglevel", "error", "pipe:1"],
        capture_output=True, check=True,
    ).stdout
    audio = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).copy()
    sr = target_sr
    wav = torch.from_numpy(audio.T).unsqueeze(0).to(device)  # [1, 2, samples]
    load_time = time.perf_counter() - t0
    duration = wav.shape[-1] / target_sr
    print(f"Audio loaded: {duration:.1f}s, resample+load {load_time:.1f}s")

    # Separate
    t0 = time.perf_counter()
    autocast = (torch.autocast("cuda", dtype=torch.float16) if fp16 and device == "cuda"
                else contextlib.nullcontext())
    with torch.no_grad(), autocast:
        sources = apply_model(model, wav, device=device, overlap=overlap)
    sep_time = time.perf_counter() - t0
    print(f"Separation: {sep_time:.1f}s ({duration/sep_time:.1f}x realtime)")

    # Save stems as opus. PCM is piped straight into ffmpeg, so no wav ever
    # touches the disk: a 6-stem 5.5 min track is 330 MB as wav against 22 MB
    # here, and an intermediate wav would just be deleted anyway.
    #
    # Encoded in parallel: libopus is single-threaded, so six sequential
    # encodes took 26.6s against 6.4s concurrent — four times longer than the
    # separation they follow.
    t0 = time.perf_counter()
    pcms = [sources[0, i].float().cpu().numpy().T.astype(np.float32)
            for i in range(len(model.sources))]

    def encode(item):
        name, pcm = item
        stem_path = os.path.join(output_dir, f"{name}.{AUDIO_EXT}")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "f32le", "-ar", str(target_sr), "-ac", "2", "-i", "pipe:0",
             "-c:a", AUDIO_CODEC, "-b:a", AUDIO_BITRATE,
             "-compression_level", AUDIO_EFFORT, stem_path],
            input=pcm.tobytes(), check=True,
        )
        return name, stem_path

    with ThreadPoolExecutor(max_workers=len(pcms)) as pool:
        stems = dict(pool.map(encode, zip(model.sources, pcms)))
    for name, path in stems.items():
        print(f"  {name}: {path}")
    print(f"Encoded {len(stems)} stems: {time.perf_counter() - t0:.1f}s")

    return stems, {"duration": duration, "sep_time": sep_time,
                    "rtf": duration / sep_time, "device": device}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python separate.py input.mp3 [output_dir] [model]")
        sys.exit(1)
    stems, stats = separate(
        sys.argv[1],
        sys.argv[2] if len(sys.argv) > 2 else None,
        sys.argv[3] if len(sys.argv) > 3 else MODEL,
    )
    print(json.dumps({"stems": stems, "stats": stats}, indent=2))
