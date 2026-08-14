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
- **A signal handler cannot interrupt C code.** SIGINT/SIGTERM raise
  `KeyboardInterrupt`, but Python only runs the handler once the interpreter is
  back from `mlx_whisper.transcribe`, so a stop lands at the *end* of the
  segment in flight, not immediately. Don't add a timeout expecting instant
  death; a long segment is seconds.

## Caches & re-running

Everything in `work/` is a cache. `*.diar.json` is the expensive one (~CPU
minutes); it's kept so re-runs skip diarization. To fix a mis-detected
language, edit `work/<base>.speaker_lang.json` and re-run — only transcription
re-runs, fast. Delete a cache file to force that step again.

`*.partial.jsonl` is the exception: it is written *during* step 4, one line per
finished segment, and deleted when the run completes. It only exists while
there is something to resume, so its presence means the last run was killed.
Records are keyed on `(start, end, speaker)` — a re-diarization that moves the
boundaries therefore misses every stale record instead of matching the wrong
one, which is why deleting `.diar.json` does not require deleting this too.

Measured on an Apple M5 (macOS 26.6.1), 10m54s of two-speaker Norwegian audio,
weights already downloaded:

```
tid: lyd 0s · diarization 4m37s · språk 5s · transkribering 1m17s
```

6m02s total, **76 % of it diarization**. Don't reason about the ratio from
transcription speed — step 2 is CPU-bound by choice, step 4 is not.

## Progress output has two modes

Default is the human-readable text that has always been there. `--progress
json` emits one JSON object per line on stdout and *nothing else* — that mode
exists so [Oschlo/schous](https://github.com/Oschlo/schous) does not have to
regex prose off two streams. If you add a `print()` to the pipeline, guard it
on `PROGRESS != "json"` or you break the frontend's parser silently.

pyannote's own `ProgressHook` is `rich`-based and writes to **stdout**, so it
cannot coexist with JSON mode; the hook is a plain callable and JSON mode
passes its own.

## Known limitation

Diarization can over-split one speaker into several IDs (e.g. one Norwegian
speaker became SPEAKER_00/01/03). Transcription text is still correct; only the
labels fragment. Fix by merging labels in the output, or re-run with
`--speakers N` if the true count is known.

## Folders

`input/` (mp4), `work/` (intermediates), `output/` (txt/srt/json). All three
are git-ignored — never commit media or transcripts. Code lives at the root.
