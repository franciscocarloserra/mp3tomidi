# mp3tomidi

Weights: `serve.py` loads `MUSCRIPTOR_WEIGHTS` (default `large`, downloaded from HuggingFace `MuScriptor/muscriptor-large` on first run into `~/.cache/muscriptor/`; the HF license must be accepted and `hf auth login` done). A local `.safetensors` path also works.

Demucs runs as a subprocess (`DEMUCS_PYTHON separate.py <file> <out>`) so its VRAM is freed after each split.

The static demo is `web/demo/<song>/` (a cached `/convert` stream + vocals stem + .mid per song, listed in `DEMO_SONGS`); `.github/workflows/pages.yml` publishes `web/` to GitHub Pages on push to main.
