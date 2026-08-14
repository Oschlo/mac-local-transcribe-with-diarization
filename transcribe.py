#!/usr/bin/env python3
"""Lokal transkripsjon med diarization og språk-låsing per taler.

Bruk:
    python transcribe.py "input/fil.mp4" [--speakers N]

Mellomresultater caches i work/ (slett dem for å kjøre på nytt):
    work/fil.wav                 16 kHz mono lyd
    work/fil.diar.json           diarization (tregt steget — caches alltid)
    work/fil.speaker_lang.json   {taler: språk} — REDIGERBAR, leses ved ny kjøring

Output i output/:
    output/fil.txt  output/fil.srt  output/fil.json
"""
import sys, os, json, subprocess, argparse, time
import numpy as np
import soundfile as sf

MODEL = "mlx-community/whisper-large-v3-mlx"
WORK_DIR = "work"      # mellomresultater: wav, diarization-cache, språk-map
OUTPUT_DIR = "output"  # leveranser: txt, srt, json
SR = 16000
MERGE_GAP = 0.75   # slå sammen nabosegmenter fra samme taler med mindre opphold (s)
MIN_SEG = 0.4      # hopp over segmenter kortere enn dette (s)


def extract_audio(src, wav):
    if os.path.exists(wav):
        return
    cmd = ["ffmpeg", "-y", "-i", src, "-ar", str(SR), "-ac", "1",
           "-c:a", "pcm_s16le", wav]
    try:
        # stderr fanges, ikke kastes: den er den eneste diagnostikken ffmpeg gir.
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE, text=True)
    except FileNotFoundError:
        sys.exit("ffmpeg ikke funnet på PATH. brew install ffmpeg\n"
                 f"  PATH={os.environ.get('PATH', '')}")
    if r.returncode:
        sys.exit(f"ffmpeg feilet på {src} (kode {r.returncode}):\n"
                 + (r.stderr or "").strip()[-800:])


def diarize(wav, cache, num_speakers=None):
    if os.path.exists(cache):
        return json.load(open(cache))
    import torch
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.utils.hook import ProgressHook
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN ikke satt. export HF_TOKEN=hf_...")
    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1",
                                    token=token)
    # ponytail: CPU. pyannote+MPS har hatt korrekthetsfeil; bytt til mps hvis for tregt.
    pipe.to(torch.device("cpu"))
    with ProgressHook() as hook:  # innebygd fremdrift per understeg (segm. → embeddings → clustering)
        dia = pipe(wav, num_speakers=num_speakers, hook=hook)
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
    items = sorted(longest.items())
    for i, (spk, (s, _)) in enumerate(items, 1):
        clip = audio[int(s["start"] * SR):int(s["end"] * SR)]
        r = mlx_whisper.transcribe(clip, path_or_hf_repo=MODEL)
        mapping[spk] = r.get("language", "no")
        print(f"  taler {i}/{len(items)} {spk}: {mapping[spk]}  (\"{r['text'].strip()[:60]}...\")")
    json.dump(mapping, open(cache, "w"), indent=2)
    print(f"\n  -> språk skrevet til {cache} — REDIGER der hvis no/sv er forvekslet, slett {os.path.basename(cache)} for ny autodeteksjon.")
    return mapping


def transcribe_segments(audio, segs, langs):
    import mlx_whisper
    from tqdm import tqdm
    out = []
    # mininterval: tett oppdatering i terminal, sjelden i fil/bakgrunn (unngår tusenvis av loggrader)
    bar = tqdm(segs, unit="seg", desc="  transkriberer",
               mininterval=0.5 if sys.stderr.isatty() else 10)
    for s in bar:
        if s["end"] - s["start"] < MIN_SEG:
            continue
        clip = audio[int(s["start"] * SR):int(s["end"] * SR)]
        lang = langs.get(s["speaker"], "no")
        r = mlx_whisper.transcribe(clip, path_or_hf_repo=MODEL, language=lang)
        text = r["text"].strip()
        if text:
            out.append({**s, "language": lang, "text": text})
        bar.set_postfix_str(f"{s['speaker']} {lang}")
    return out


def fmt_dur(sec):
    m, s = divmod(int(sec), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--speakers", type=int, default=None, help="antall talere hvis kjent")
    args = ap.parse_args()

    base = os.path.splitext(os.path.basename(args.src))[0]
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    wav = os.path.join(WORK_DIR, base + ".wav")
    out_base = os.path.join(OUTPUT_DIR, base)

    timings = {}

    print("1/4 lyd…")
    t = time.monotonic()
    extract_audio(args.src, wav)
    timings["lyd"] = time.monotonic() - t

    print("2/4 diarization…")
    t = time.monotonic()
    segs = merge_segments(diarize(wav, os.path.join(WORK_DIR, base + ".diar.json"), args.speakers))
    timings["diarization"] = time.monotonic() - t
    print(f"  {len(segs)} segmenter, {len(set(s['speaker'] for s in segs))} talere")

    audio = sf.read(wav, dtype="float32")[0]
    print("3/4 språk per taler…")
    t = time.monotonic()
    langs = detect_languages(audio, segs, os.path.join(WORK_DIR, base + ".speaker_lang.json"))
    timings["språk"] = time.monotonic() - t

    print("4/4 transkriberer…")
    t = time.monotonic()
    out = transcribe_segments(audio, segs, langs)
    timings["transkribering"] = time.monotonic() - t

    write_outputs(out, out_base)
    print(f"\nFerdig: {out_base}.txt / .srt / .json")
    print("  tid: " + " · ".join(f"{k} {fmt_dur(v)}" for k, v in timings.items()))


def _selfcheck():
    assert ts(3661.5) == "01:01:01,500", ts(3661.5)
    assert merge_segments([
        {"start": 0, "end": 1, "speaker": "A"},
        {"start": 1.2, "end": 2, "speaker": "A"},
        {"start": 5, "end": 6, "speaker": "A"},
    ]) == [{"start": 0, "end": 2, "speaker": "A"}, {"start": 5, "end": 6, "speaker": "A"}]

    # en ffmpeg-feil skal bli en lesbar melding, ikke en tom CalledProcessError
    try:
        extract_audio("finnes-ikke-ffb4e1.mp4", "/tmp/finnes-ikke-ffb4e1.wav")
    except SystemExit as e:
        assert "ffmpeg" in str(e), e
    else:
        assert False, "extract_audio skulle avsluttet på manglende input"

    print("selfcheck ok")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selfcheck":
        _selfcheck()
    else:
        main()
