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
- Python 3.12+, driven by [`uv`](https://github.com/astral-sh/uv)
  (`brew install uv`)
- **~4.2 GB of disk** — see the table below
- A Hugging Face account and a read token

## Setup

**Accept the two model licenses first, before the first run.** Both are gated,
and both must be accepted with the *same* account the token belongs to — an
easy thing to get wrong if you have two:

- https://huggingface.co/pyannote/speaker-diarization-community-1
- https://huggingface.co/pyannote/segmentation-3.0

Skip this and step 2 fails with a Hugging Face download error that does not
mention licenses at all. The token itself is valid, so it looks like something
else is wrong. It is the most likely first failure anyone hits.

```zsh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt

.venv/bin/hf auth login    # paste your read token
```

`hf auth login` stores the token in `~/.cache/huggingface/token`, which is where
`huggingface_hub.get_token()` looks. `export HF_TOKEN=hf_...` still works and
still wins over the file — but only for processes that inherit your shell. A GUI
frontend launched from Finder inherits nothing, so the file is the one that
works everywhere.

### Disk

Measured 2026-08-12:

| What | Size |
|---|---|
| `mlx-community/whisper-large-v3-mlx` | **2.9 GB** |
| `pyannote/speaker-diarization-community-1` | 31 MB |
| `.venv` (torch and torchaudio are almost all of it) | **1.3 GB** |
| total | **~4.2 GB** |

The weights are **not** downloaded by `uv pip install` — they arrive during the
**first run**, in the middle of steps 2 and 4. So the first job looks like it
has hung while it is in fact pulling 2.9 GB, and if you piped stdout to a file
there is nothing in the log saying so.

### How long it takes

Measured on an **Apple M5, macOS 26.6.1**, weights already downloaded, a
10m54s two-speaker Norwegian recording:

```
tid: lyd 0s · diarization 4m37s · språk 5s · transkribering 1m17s
```

**6m02s total for 10m54s of audio**, about 0.55× the length of the recording —
and **diarization is 76 % of it**. That ratio is not obvious, so do not
extrapolate from transcription speed: diarization runs on CPU on purpose, which
is why it dominates. `lyd 0s` is an already-16 kHz WAV input; an `.mp4` adds a
few seconds of ffmpeg.

The script prints that timing line at the end of every run, so you can measure
your own machine rather than trust this one.

Re-running the same file is much faster: steps 1–3 are cached in `work/`, so
only transcription runs again (1m17s of the 6m02s above).

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

# somewhere other than ./work and ./output:
.venv/bin/python transcribe.py "input/recording.mp4" \
    --work-dir /tmp/w --output-dir ~/Transcripts

# one JSON line per event on stdout, for a frontend:
.venv/bin/python transcribe.py "input/recording.mp4" --progress json
```

Each step reports progress: diarization shows per-substep bars, language
detection shows `taler i/N`, the transcription loop is a percent/ETA bar, and a
per-step timing summary prints at the end. In a real terminal the bars update
live; piped to a file they back off to occasional lines.

`--progress json` replaces all of that with one JSON object per line on stdout
and nothing else — `step`, `progress`, `diarized`, `language`, `resume`,
`done`, `interrupted` — so a GUI does not have to scrape prose off two streams.
It also drops the 10-second backoff, since there is no bar to repaint.

### Interrupting a run

Ctrl-C, or a `SIGTERM`, stops after the segment in flight and still writes
`.txt`/`.srt`/`.json` from everything transcribed so far (exit code 130 or
143). Nothing is lost: each finished segment is appended to
`work/<name>.partial.jsonl` as it completes, and re-running the same file picks
up from there. The partial file is deleted once a run completes.

After the first run, set `HF_HUB_OFFLINE=1` to guarantee nothing touches the
network.

### Outputs

| File | What |
|------|------|
| `work/recording.diar.json` | diarization result (cached; delete to re-diarize) |
| `work/recording.speaker_lang.json` | `{speaker: language}` — **edit this** if a language was mis-detected, then re-run |
| `work/recording.partial.jsonl` | segments finished by an interrupted run; only exists while there is something to resume |
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

## License

MIT — see [LICENSE](LICENSE).

## Third-party

Nothing in the dependency tree constrains the license above, but two points are
easy to get wrong:

**`ffmpeg` does not affect this license.** It is called as a subprocess
(`transcribe.py`), never linked, and you install it yourself with Homebrew. A
GPL-built ffmpeg binary on your machine stays your ffmpeg binary.

**No model weights are distributed from here.** Both pyannote models are gated
on Hugging Face: every user accepts the terms with their own account and
downloads with their own token. This repo contains no weights and cannot route
around that gate. The CC-BY-4.0 attribution below therefore applies to your use
of `speaker-diarization-community-1`, not to anything shipped in this
repository.

| Component | License |
|---|---|
| `pyannote.audio` (code) | MIT |
| `pyannote/speaker-diarization-community-1` (weights) | **CC-BY-4.0** — requires attribution |
| `pyannote/segmentation-3.0` (weights) | MIT |
| `mlx-whisper` | MIT |
| `mlx-community/whisper-large-v3-mlx` (weights) | MIT |
| `torch`, `torchaudio`, `numpy`, `soundfile` | BSD-3-Clause |
| `ffmpeg` | GPL/LGPL depending on build — subprocess, not linked |
