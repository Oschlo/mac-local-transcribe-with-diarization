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

## Folder layout

```
input/    your *.mp4 source videos
work/     intermediates: *.wav, *.diar.json, *.speaker_lang.json
output/   deliverables:  *.txt, *.srt, *.json
```

`work/` and `output/` are created automatically. All three are git-ignored.

## Usage

Put videos in `input/`, then:

```zsh
.venv/bin/python transcribe.py "input/recording.mp4"

# if you know the speaker count, it improves diarization:
.venv/bin/python transcribe.py "input/recording.mp4" --speakers 2

# all of them:
for f in input/*.mp4; do .venv/bin/python transcribe.py "$f"; done
```

Each step reports progress: diarization shows per-substep bars, language
detection shows `taler i/N`, the transcription loop is a percent/ETA bar, and a
per-step timing summary prints at the end. In a real terminal the bars update
live; piped to a file they back off to occasional lines.

After the first run, set `HF_HUB_OFFLINE=1` to guarantee nothing touches the
network.

### Outputs

| File | What |
|------|------|
| `work/recording.diar.json` | diarization result (cached; delete to re-diarize) |
| `work/recording.speaker_lang.json` | `{speaker: language}` — **edit this** if a language was mis-detected, then re-run |
| `output/recording.txt` | readable transcript, one line per segment with speaker + language + timestamp |
| `output/recording.srt` | subtitles |
| `output/recording.json` | full structured output (start, end, speaker, language, text) |

Example `.txt`:

```
[00:00:04] SPEAKER_00 (sv): Väldigt många människor som ska försöka förstå sig på det.
[00:00:38] SPEAKER_01 (no): Nå er du inne på veldig mye spennende her.
```

### Fixing a mis-detected language

If step 3 maps a speaker to the wrong language, edit `work/recording.speaker_lang.json`:

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
