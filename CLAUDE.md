# CLAUDE.md

Local diarized transcription for NO/SV-mixed recordings on Apple Silicon.
One script, `transcribe.py`. Read the README for usage; this file is the
non-obvious stuff that bit us once already.

## Run it

```zsh
HF_TOKEN=hf_... .venv/bin/python transcribe.py "input/foo.mp4" [--speakers N]
```

Use the venv (`.venv/bin/python`), not system Python. `HF_TOKEN` must be set
even with `HF_HUB_OFFLINE=1` (the script checks for it).

## Hard-won gotchas

- **pyannote must be 4.x, model is `speaker-diarization-community-1`.** Don't
  "fix" it back to `speaker-diarization-3.1`: pyannote 3.x is incompatible with
  current torchaudio (`torchaudio.AudioMetaData` was removed → ImportError), and
  4.x re-points the 3.1 alias at community-1 anyway. `community-1` is gated —
  the HF account must accept its license (plus `pyannote/segmentation-3.0`).
- **pyannote 4.x pipeline returns a `DiarizeOutput`**, not an `Annotation`. Use
  `dia.serialize()["diarization"]`, not `.itertracks()`.
- **Diarization runs on CPU on purpose** (`pipe.to("cpu")`). MPS has had
  correctness bugs with pyannote. Slow but right; flip to MPS only if you
  measure and verify.
- **Transcription is mlx-whisper, not whisperx.** whisperx falls back to slow
  CPU on Apple Silicon. mlx uses MLX and is fast on M-series.
- **Language is locked per speaker, never autodetected during the real run.**
  That's the whole point — Whisper otherwise mistranslates NO↔SV. Step 3
  autodetects once per speaker; step 4 forces `language=`.
- **Python buffers stdout when piped to a file**, so `print()` progress looks
  frozen in background logs. tqdm writes to stderr (`\r`, unbuffered) and is the
  reliable live signal. This is why a backgrounded run can look dead but isn't —
  check the process is using CPU (`ps -o %cpu`).

## Caches & re-running

Everything in `work/` is a cache. `*.diar.json` is the expensive one (~CPU
minutes); it's kept so re-runs skip diarization. To fix a mis-detected
language, edit `work/<base>.speaker_lang.json` and re-run — only transcription
re-runs, fast. Delete a cache file to force that step again.

## Known limitation

Diarization can over-split one speaker into several IDs (e.g. one Norwegian
speaker became SPEAKER_00/01/03). Transcription text is still correct; only the
labels fragment. Fix by merging labels in the output, or re-run with
`--speakers N` if the true count is known.

## Folders

`input/` (mp4), `work/` (intermediates), `output/` (txt/srt/json). All three
are git-ignored — never commit media or transcripts. Code lives at the root.
