#!/usr/bin/env python3
"""Transkripsjon med valgbar motor.

Motorer (--engine):
    mlx     (standard) lokal/offline: pyannote-diarisering + mlx-whisper med
            språk-låsing per taler (NO<->SV). Krever HF-token + gated modeller.
    soniox  sky: norsk-tunet, ingen HF-token. Talere skilles enten via Soniox'
            egen sky-diarisering, eller via --dual (én mono-fil per taler).

Bruk:
    python transcribe.py "fil.mp4" [--speakers N]                # mlx (lokal)
    python transcribe.py "fil.mp4" --engine soniox               # sky-diarisering
    python transcribe.py out.md --engine soniox --dual motpart.wav meg.wav

Mellomresultater (kun mlx) caches ved siden av input (slett for ny kjøring):
    fil.wav                 16 kHz mono lyd
    fil.diar.json           diarization (tregt steget — caches alltid)
    fil.speaker_lang.json   {taler: språk} — REDIGERBAR, leses ved ny kjøring

Output (begge motorer):
    fil.txt  fil.srt  fil.json
"""
import sys, os, json, subprocess, argparse
import numpy as np
import soundfile as sf

MODEL = "mlx-community/whisper-large-v3-mlx"
SR = 16000
MERGE_GAP = 0.75   # slå sammen nabosegmenter fra samme taler med mindre opphold (s)
MIN_SEG = 0.4      # hopp over segmenter kortere enn dette (s)


def extract_audio(src, wav):
    if os.path.exists(wav):
        return
    subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-ar", str(SR), "-ac", "1",
         "-c:a", "pcm_s16le", wav],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def diarize(wav, cache, num_speakers=None):
    if os.path.exists(cache):
        return json.load(open(cache))
    import torch
    from pyannote.audio import Pipeline
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN ikke satt. export HF_TOKEN=hf_...")
    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1",
                                    token=token)
    # ponytail: CPU. pyannote+MPS har hatt korrekthetsfeil; bytt til mps hvis for tregt.
    pipe.to(torch.device("cpu"))
    dia = pipe(wav, num_speakers=num_speakers)
    segs = dia.serialize()["diarization"]
    json.dump(segs, open(cache, "w"), indent=2)
    return segs


def merge_segments(segs):
    """Slå sammen sammenhengende segmenter fra samme taler med kort opphold."""
    out = []
    for s in sorted(segs, key=lambda x: x["start"]):
        if out and out[-1]["speaker"] == s["speaker"] and s["start"] - out[-1]["end"] <= MERGE_GAP:
            out[-1]["end"] = s["end"]
        else:
            out.append(dict(s))
    return out


def detect_languages(audio, segs, cache):
    """Bygg {taler: språk} fra det lengste segmentet per taler. Redigerbar cache."""
    if os.path.exists(cache):
        return json.load(open(cache))
    import mlx_whisper
    longest = {}
    for s in segs:
        spk, dur = s["speaker"], s["end"] - s["start"]
        if dur > longest.get(spk, (None, 0))[1]:
            longest[spk] = (s, dur)
    mapping = {}
    for spk, (s, _) in sorted(longest.items()):
        clip = audio[int(s["start"] * SR):int(s["end"] * SR)]
        r = mlx_whisper.transcribe(clip, path_or_hf_repo=MODEL)
        mapping[spk] = r.get("language", "no")
        print(f"  {spk}: {mapping[spk]}  (\"{r['text'].strip()[:60]}...\")")
    json.dump(mapping, open(cache, "w"), indent=2)
    print(f"\n  -> språk skrevet til {cache} — REDIGER der hvis no/sv er forvekslet, slett {os.path.basename(cache)} for ny autodeteksjon.")
    return mapping


def transcribe_segments(audio, segs, langs):
    import mlx_whisper
    out = []
    for i, s in enumerate(segs, 1):
        if s["end"] - s["start"] < MIN_SEG:
            continue
        clip = audio[int(s["start"] * SR):int(s["end"] * SR)]
        lang = langs.get(s["speaker"], "no")
        r = mlx_whisper.transcribe(clip, path_or_hf_repo=MODEL, language=lang)
        text = r["text"].strip()
        if text:
            out.append({**s, "language": lang, "text": text})
        print(f"  [{i}/{len(segs)}] {s['speaker']} ({lang}) {text[:60]}")
    return out


def ts(sec, sep=","):
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    ms = int((sec - int(sec)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def write_outputs(segs, base):
    json.dump(segs, open(base + ".json", "w"), ensure_ascii=False, indent=2)
    with open(base + ".txt", "w") as f:
        for s in segs:
            f.write(f"[{ts(s['start'], '.')[:-4]}] {s['speaker']} ({s['language']}): {s['text']}\n")
    with open(base + ".srt", "w") as f:
        for i, s in enumerate(segs, 1):
            f.write(f"{i}\n{ts(s['start'])} --> {ts(s['end'])}\n"
                    f"{s['speaker']} ({s['language']}): {s['text']}\n\n")


def run_mlx(args, base):
    wav = base + ".wav"
    print("1/4 lyd…")
    extract_audio(args.src, wav)
    print("2/4 diarization…")
    segs = merge_segments(diarize(wav, base + ".diar.json", args.speakers))
    print(f"  {len(segs)} segmenter, {len(set(s['speaker'] for s in segs))} talere")

    audio = sf.read(wav, dtype="float32")[0]
    print("3/4 språk per taler…")
    langs = detect_languages(audio, segs, base + ".speaker_lang.json")
    print("4/4 transkriberer…")
    return transcribe_segments(audio, segs, langs)


def run_soniox(args, base):
    import soniox_engine
    if args.dual:
        if len(args.dual) != 2:
            sys.exit("--dual tar nøyaktig to filer: <motpart.wav> <meg.wav>")
        print("soniox (dual-kanal)…")
        return soniox_engine.transcribe_dual(
            [(args.dual[0], "SPEAKER_00"), (args.dual[1], "SPEAKER_01")],
            language=args.language)
    print("soniox (sky-diarisering)…")
    return soniox_engine.transcribe_diarized(args.src, language=args.language)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="lydfil (mlx/soniox), eller output-basenavn ved --dual")
    ap.add_argument("--engine", choices=["mlx", "soniox"], default="mlx",
                    help="transkriberings-motor (standard: mlx, lokal)")
    ap.add_argument("--speakers", type=int, default=None,
                    help="antall talere hvis kjent (kun mlx)")
    ap.add_argument("--dual", nargs=2, metavar=("MOTPART", "MEG"),
                    help="soniox: to mono-filer, én taler per fil")
    ap.add_argument("--language", default="no",
                    help="soniox språk-hint (kun mlx låser språk per taler)")
    args = ap.parse_args()

    base = os.path.splitext(args.src)[0]
    out = run_soniox(args, base) if args.engine == "soniox" else run_mlx(args, base)
    write_outputs(out, base)
    print(f"\nFerdig: {base}.txt / .srt / .json")


def _selfcheck():
    assert ts(3661.5) == "01:01:01,500", ts(3661.5)
    assert merge_segments([
        {"start": 0, "end": 1, "speaker": "A"},
        {"start": 1.2, "end": 2, "speaker": "A"},
        {"start": 5, "end": 6, "speaker": "A"},
    ]) == [{"start": 0, "end": 2, "speaker": "A"}, {"start": 5, "end": 6, "speaker": "A"}]
    print("selfcheck ok")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selfcheck":
        _selfcheck()
    else:
        main()
