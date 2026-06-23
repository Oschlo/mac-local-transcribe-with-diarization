# mac-local-transcribe-with-diarization

Local, offline transcription of multi-speaker recordings on Apple Silicon, with
speaker diarization and **per-speaker language locking** — built for conversations
where one speaker stays in Norwegian and another in Swedish (or any fixed
language per speaker).

Everything runs locally on an M1/M2/M3 Mac. The only network access is a
one-time download of model weights from Hugging Face.

## Why language locking

Whisper happily mis-detects or "translates" between closely related languages
(Norwegian ↔ Swedish). This pipeline sidesteps that: it figures out each
speaker's language once, then forces `language=` explicitly for every segment
that speaker produces. No autodetection during the real run.

## How it works

1. **Extract** 16 kHz mono WAV from the video (ffmpeg).
2. **Diarize** with pyannote `speaker-diarization-community-1` → segments of
   `(start, end, speaker)`.
3. **Detect language per speaker** — transcribe each speaker's longest segment
   in autodetect mode, build a `{speaker: language}` map. Editable.
4. **Transcribe** each segment with [mlx-whisper](https://github.com/ml-explore/mlx-examples)
   (`large-v3`), `language` locked to the speaker's language.
5. **Merge** into chronological `.txt`, `.srt`, and `.json`.

Intermediate results are cached next to the input so you can re-run later steps
without re-diarizing.

## Requirements

- macOS on Apple Silicon
- `ffmpeg` (`brew install ffmpeg`)
- Python 3.12+ (this repo uses [`uv`](https://github.com/astral-sh/uv))
- A Hugging Face account + token, and **accepted licenses** for two gated models:
  - https://huggingface.co/pyannote/speaker-diarization-community-1
  - https://huggingface.co/pyannote/segmentation-3.0

## Setup

```zsh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt

export HF_TOKEN="hf_..."   # your read token
```

## Usage

```zsh
.venv/bin/python transcribe.py "recording.mp4"

# if you know the speaker count, it improves diarization:
.venv/bin/python transcribe.py "recording.mp4" --speakers 2
```

After the first run, set `HF_HUB_OFFLINE=1` to guarantee nothing touches the
network.

## Transcription engines

Two engines, selected with `--engine`:

| Engine | Where | Speaker split | Language | HF token |
|--------|-------|---------------|----------|----------|
| `mlx` (default) | local / offline | pyannote diarization | **per-speaker locking** (NO↔SV) | required |
| `soniox` | cloud | Soniox cloud diarization, or one channel per speaker | Norwegian-tuned, single language | not needed |

Use `mlx` when you need offline processing or per-speaker language locking
(the original use case). Use `soniox` for Norwegian recordings/calls where you
want a fast cloud transcription without the gated HF models — or for **live**
transcription (see `stream.py`).

### Soniox setup

```zsh
pip install soniox                      # SDK (only needed for soniox / stream.py)
mkdir -p ~/.soniox && echo "YOUR_KEY" > ~/.soniox/api-key   # or export SONIOX_API_KEY
```

### Soniox — batch (a recording → transcript)

```zsh
# mixed recording → Soniox cloud diarization (no HF token):
.venv/bin/python transcribe.py "recording.mp4" --engine soniox

# two mono files, one known speaker each (must share a clock / recorded together):
.venv/bin/python transcribe.py out.md --engine soniox --dual motpart.wav meg.wav
```

Outputs are the same `.txt` / `.srt` / `.json` as the `mlx` engine.

### Soniox — realtime (live call)

`stream.py` transcribes a live Mac call as it happens. Two sources, each its own
Soniox realtime session, so no diarization is needed:

- `helen_systemtap` (bundled Swift binary) captures the **other party**
- the **microphone** (ffmpeg / avfoundation) captures **you**

```zsh
.venv/bin/python stream.py [out.md]     # prints live; Ctrl-C writes the merged transcript
```

macOS-only (CoreAudio system tap). Override the mic with `STREAM_MIC_DEVICE` and
the binary path with `HELEN_SYSTEMTAP`. Rebuild the tap from
`helen_systemtap.swift` if needed.

### Outputs (next to the input file)

| File | What |
|------|------|
| `recording.diar.json` | diarization result (cached; delete to re-diarize) |
| `recording.speaker_lang.json` | `{speaker: language}` — **edit this** if a language was mis-detected, then re-run |
| `recording.txt` | readable transcript, one line per segment with speaker + language + timestamp |
| `recording.srt` | subtitles |
| `recording.json` | full structured output (start, end, speaker, language, text) |

Example `.txt`:

```
[00:00:04] SPEAKER_00 (sv): Väldigt många människor som ska försöka förstå sig på det.
[00:00:38] SPEAKER_01 (no): Nå er du inne på veldig mye spennende her.
```

### Fixing a mis-detected language

If step 3 maps a speaker to the wrong language, edit `recording.speaker_lang.json`:

```json
{ "SPEAKER_00": "sv", "SPEAKER_01": "no" }
```

The diarization cache is kept, so re-running only re-transcribes — fast.

## Notes

- **Overlapping speech** (people talking over each other) is handled only
  loosely by diarization; expect some noise there.
- **Very short segments** (< 0.4 s) are skipped.
- Expect to do a light manual pass at the end. The goal is to minimize
  correction, not eliminate it.

See [transkripsjon-pipeline-plan.md](transkripsjon-pipeline-plan.md) for the
original design notes.
