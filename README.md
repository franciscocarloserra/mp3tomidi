# mp3tomidi

### Your song goes in. Notes come out. Nobody asked for this.

You have an mp3. You want a MIDI. Your piano teacher stopped answering your calls years ago. Enter **mp3tomidi**: a web panel that feeds your audio to **two** neural networks, one after the other, until the music surrenders.

- [MuScriptor](https://github.com/muscriptor/muscriptor) listens to the whole song and writes down every note of every instrument, like a stenographer who did a PhD in vibes.
- [Demucs htdemucs](https://github.com/facebookresearch/demucs) rips the singer out of the mix, then plays them back over a synth that sounds like a 1997 ringtone with ambition.

**[Live demo](https://franciscocarloserra.github.io/mp3tomidi/)**: no GPU was harmed in the making of this page. It replays one cached transcription of a 100% AI-slop song (no copyright, no shame), with its isolated vocals. Hit "split voz" and enjoy the haunting duet between a real human-ish voice and a sawtooth wave.

## The Pipeline™

1. You drop a file. The panel throws it at `serve.py` (`POST /convert`).
2. MuScriptor chews it in 5-second bites and streams every note on/off back over SSE, so you watch the piano roll fill up live, like a very slow, very nerdy fireworks show. Each instrument gets its own color and its own track button.
3. When it's done it finds the beat (beat_this), snaps a beat grid on top and bakes you a real `.mid` file you can download and show your friends, who will not care.
4. "split voz" sends the same file to `POST /voice`. `separate.py` fires up htdemucs on the GPU, extracts the vocals and hands them back. The panel lines them up with the MIDI using the onset delay from the beat grid. Karaoke, but backwards.
5. Playback goes through a WebAudio synth of questionable taste, **or** straight out over Web MIDI to your actual keyboard: one channel and one General MIDI program per instrument, drums on channel 10 like the ancients intended.

Everything is cached per file in `data/<sha1>.jsonl` (+ `.mid`, `_stems/`). Ask for the same song twice and it answers instantly, smug. Kill it halfway and it picks up from the last finished chunk, holding no grudges.

## Summoning ritual

    python -m venv .venv && .venv/bin/pip install muscriptor
    pip install demucs          # any python; point DEMUCS_PYTHON at it (default: python3)
    .venv/bin/python serve.py   # the service awakens on :6977
    cd web && python3 -m http.server 7808   # the panel, at http://localhost:7808

`MUSCRIPTOR_WEIGHTS` picks the brain: `large` (default, downloaded from HuggingFace on first run; accept the license there first) or a local `.safetensors` path. The large model eats ~11 GB of VRAM for breakfast. On an RTX 3090, 15 s of audio takes ~25 s, so yes, it is slower than just listening to the song.

## Controls for the brave

- Click the roll: play / stop.
- Ctrl+wheel or `+` / `-`: zoom in until a single note fills your monitor.
- Click a track: mute. Double click: solo. Double click again: everyone's back. Right click: that track goes to MIDI channel 1, the one your keyboard lights up.

## The robot

Yes, there is a robot. Open `http://<this-ip>:6977/robot` on your phone (same network), put the vocals on, and a neon 3D robot dances to the singer. When it gets loud, it loses its mind: arms up, strobe, sparks. We are not sorry.

## Known crimes

MuScriptor gives you notes per instrument but no velocity, so everything plays at the same level of enthusiasm. The synth is a toy, not a soundfont. The demo page can't transcribe new files, because GitHub refused to lend us a GPU.
