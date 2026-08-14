#!/usr/bin/env python3
"""Lokal transkripsjon med diarization og språk-låsing per taler.

Bruk:
    python transcribe.py "input/fil.mp4" [--speakers N]
                         [--work-dir DIR] [--output-dir DIR]

Mellomresultater caches i work/ (slett dem for å kjøre på nytt):
    work/fil.wav                 16 kHz mono lyd
    work/fil.diar.json           diarization (tregt steget — caches alltid)
    work/fil.speaker_lang.json   {taler: språk} — REDIGERBAR, leses ved ny kjøring
    work/fil.partial.jsonl       ferdige segmenter fra en avbrutt kjøring;
                                 slettes når kjøringen fullfører

Output i output/:
    output/fil.txt  output/fil.srt  output/fil.json
"""
import sys, os, json, subprocess, argparse, time, signal
import numpy as np
import soundfile as sf

MODEL = "mlx-community/whisper-large-v3-mlx"
WORK_DIR = "work"      # mellomresultater: wav, diarization-cache, språk-map
OUTPUT_DIR = "output"  # leveranser: txt, srt, json
SR = 16000
MERGE_GAP = 0.75   # slå sammen nabosegmenter fra samme taler med mindre opphold (s)
MIN_SEG = 0.4      # hopp over segmenter kortere enn dette (s)

# "text" = menneskelesbart som før, "json" = én JSON-linje per hendelse på
# stdout og ingenting annet. Settes én gang fra --progress.
PROGRESS = "text"


def jprint(**kw):
    print(json.dumps(kw, ensure_ascii=False), flush=True)


def step(n, name):
    if PROGRESS == "json":
        jprint(event="step", step=n, name=name)
    else:
        print(f"{n}/4 {name}…")


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
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN ikke satt. export HF_TOKEN=hf_...")
    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1",
                                    token=token)
    # ponytail: CPU. pyannote+MPS har hatt korrekthetsfeil; bytt til mps hvis for tregt.
    pipe.to(torch.device("cpu"))
    if PROGRESS == "json":
        # pyannotes egen ProgressHook er rich-basert og skriver til stdout —
        # den ville blandet seg med JSON-linjene. Hooken er bare en callable.
        def hook(step_name, artifact, file=None, total=None, completed=None):
            if total:
                jprint(event="progress", step=2, sub=step_name,
                       completed=completed or 0, total=total)
        dia = pipe(wav, num_speakers=num_speakers, hook=hook)
    else:
        from pyannote.audio.pipelines.utils.hook import ProgressHook
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
        if PROGRESS == "json":
            jprint(event="language", completed=i, total=len(items),
                   speaker=spk, language=mapping[spk])
        else:
            print(f"  taler {i}/{len(items)} {spk}: {mapping[spk]}  (\"{r['text'].strip()[:60]}...\")")
    json.dump(mapping, open(cache, "w"), indent=2)
    if PROGRESS != "json":
        print(f"\n  -> språk skrevet til {cache} — REDIGER der hvis no/sv er forvekslet, slett {os.path.basename(cache)} for ny autodeteksjon.")
    return mapping


def seg_key(s):
    """Identitet for et segment på tvers av kjøringer. Ikke bare start: to
    talere kan i prinsippet begynne på samme hundredel."""
    return (round(s["start"], 3), round(s["end"], 3), s["speaker"])


def read_partial(path):
    """Ferdige segmenter fra en avbrutt kjøring. Linjer som ikke er hel JSON
    ignoreres — den siste kan være halvskrevet hvis prosessen ble drept midt i
    en write()."""
    done = {}
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[seg_key(rec)] = rec
    return done


def transcribe_segments(audio, segs, langs, partial):
    import mlx_whisper
    done = read_partial(partial)
    if done:
        if PROGRESS == "json":
            jprint(event="resume", completed=len(done), total=len(segs))
        else:
            print(f"  gjenopptar: {len(done)} segmenter alt transkribert")
    out = []
    bar = None
    if PROGRESS != "json":
        from tqdm import tqdm
        # mininterval: tett oppdatering i terminal, sjelden i fil/bakgrunn (unngår tusenvis av loggrader)
        bar = tqdm(segs, unit="seg", desc="  transkriberer",
                   mininterval=0.5 if sys.stderr.isatty() else 10)
    for i, s in enumerate(bar if bar is not None else segs, 1):
        lang = langs.get(s["speaker"], "no")
        if s["end"] - s["start"] >= MIN_SEG:
            rec = done.get(seg_key(s))
            if rec is None:
                clip = audio[int(s["start"] * SR):int(s["end"] * SR)]
                r = mlx_whisper.transcribe(clip, path_or_hf_repo=MODEL, language=lang)
                rec = {**s, "language": lang, "text": r["text"].strip()}
                # også de tomme skrives: ellers transkriberes stillheten på nytt
                # ved hver gjenopptagelse. append + flush per segment — hele
                # poenget er at en drept prosess etterlater arbeidet på disk.
                with open(partial, "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
            if rec["text"]:
                out.append(rec)
        if bar is not None:
            bar.set_postfix_str(f"{s['speaker']} {lang}")
        else:
            jprint(event="progress", step=4, completed=i, total=len(segs),
                   speaker=s["speaker"], language=lang)
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


_signum = None


def _on_signal(signum, frame):
    """SIGTERM oppfører seg som Ctrl-C, og vi husker hvilket signal det var for
    exit-koden. Python utsetter handleren til tolkeren er tilbake fra C-kode, så
    avbruddet lander tidligst når inneværende segment er ferdig."""
    global _signum
    _signum = signum
    raise KeyboardInterrupt


def main():
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--speakers", type=int, default=None, help="antall talere hvis kjent")
    ap.add_argument("--work-dir", default=WORK_DIR, help="mellomresultater (default: %(default)s)")
    ap.add_argument("--output-dir", default=OUTPUT_DIR, help="leveranser (default: %(default)s)")
    ap.add_argument("--progress", choices=("text", "json"), default="text",
                    help="json: én JSON-linje per hendelse på stdout (default: %(default)s)")
    args = ap.parse_args()

    global PROGRESS
    PROGRESS = args.progress

    base = os.path.splitext(os.path.basename(args.src))[0]
    work_dir, output_dir = args.work_dir, args.output_dir
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    wav = os.path.join(work_dir, base + ".wav")
    out_base = os.path.join(output_dir, base)

    timings = {}

    step(1, "lyd")
    t = time.monotonic()
    extract_audio(args.src, wav)
    timings["lyd"] = time.monotonic() - t

    step(2, "diarization")
    t = time.monotonic()
    segs = merge_segments(diarize(wav, os.path.join(work_dir, base + ".diar.json"), args.speakers))
    timings["diarization"] = time.monotonic() - t
    n_spk = len(set(s["speaker"] for s in segs))
    if PROGRESS == "json":
        jprint(event="diarized", segments=len(segs), speakers=n_spk)
    else:
        print(f"  {len(segs)} segmenter, {n_spk} talere")

    audio = sf.read(wav, dtype="float32")[0]
    step(3, "språk per taler")
    t = time.monotonic()
    langs = detect_languages(audio, segs, os.path.join(work_dir, base + ".speaker_lang.json"))
    timings["språk"] = time.monotonic() - t

    step(4, "transkriberer")
    t = time.monotonic()
    partial = os.path.join(work_dir, base + ".partial.jsonl")
    try:
        out = transcribe_segments(audio, segs, langs, partial)
    except KeyboardInterrupt:
        # partial-fila er fasiten — det som står der er alt som rakk å bli
        # ferdig, uansett hvor i loopen avbruddet traff.
        out = sorted((r for r in read_partial(partial).values() if r["text"]),
                     key=lambda r: r["start"])
        write_outputs(out, out_base)
        if PROGRESS == "json":
            jprint(event="interrupted", segments=len(out), output=out_base,
                   resumable=True)
        else:
            print(f"\nAvbrutt — skrev {len(out)} segmenter til {out_base}.txt/.srt/.json.")
            print("  kjør på nytt med samme fil for å fortsette der den slapp.")
        raise
    timings["transkribering"] = time.monotonic() - t

    write_outputs(out, out_base)
    if os.path.exists(partial):
        os.remove(partial)   # kjøringen fullførte; ingenting å gjenoppta
    if PROGRESS == "json":
        jprint(event="done", segments=len(out), output=out_base,
               timings={k: round(v, 1) for k, v in timings.items()})
    else:
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

    # gjenopptagelse: nøkkelen treffer eget segment, bommer på naboen, og en
    # halvskrevet siste linje (drept midt i en write) skal ikke velte lesningen
    p = "/tmp/_sc_partial_ffb4e1.jsonl"
    with open(p, "w") as f:
        f.write(json.dumps({"start": 1.0, "end": 2.0, "speaker": "A", "text": "hei"}) + "\n")
        f.write('{"start": 3.0, "end":')
    done = read_partial(p)
    assert len(done) == 1, done
    assert done[seg_key({"start": 1.0, "end": 2.0, "speaker": "A"})]["text"] == "hei"
    assert seg_key({"start": 1.0, "end": 2.0, "speaker": "B"}) not in done
    os.remove(p)

    # SIGTERM må oppføre seg som Ctrl-C, ellers dør steg 4 uten å skrive noe
    global _signum
    try:
        _on_signal(signal.SIGTERM, None)
    except KeyboardInterrupt:
        assert _signum == signal.SIGTERM
    else:
        assert False, "_on_signal skulle ha kastet KeyboardInterrupt"
    _signum = None

    # Alt over er ren logikk. Under er det som faktisk står i veien for en ny
    # bruker: torch, pyannote og mlx_whisper importeres inne i funksjoner, så
    # et halvt installert venv passerer hele resten av denne testen.
    import shutil
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg ikke funnet på PATH. brew install ffmpeg\n"
                 f"  PATH={os.environ.get('PATH', '')}")
    for mod in ("torch", "pyannote.audio", "mlx_whisper"):
        try:
            __import__(mod)
        except ImportError as e:
            sys.exit(f"{mod} kan ikke importeres: {e}\n"
                     "  uv pip install --python .venv/bin/python -r requirements.txt")

    print("selfcheck ok")


def _check_access():
    """Det --selfcheck ikke kan svare på uten nett: er tokenet i live, og er
    modell-lisensene godtatt med kontoen det tilhører. De to feilene er
    forskjellige og skal ikke se like ut."""
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        sys.exit("HF_TOKEN ikke satt.")
    from huggingface_hub import HfApi
    from huggingface_hub.utils import (GatedRepoError, RepositoryNotFoundError,
                                       HfHubHTTPError)
    api = HfApi()
    try:
        who = api.whoami(token=tok)
    except Exception as e:
        sys.exit(f"Hugging Face avviste tokenet: {e}\n"
                 "  lag et nytt read-token på https://huggingface.co/settings/tokens")
    for repo in ("pyannote/speaker-diarization-community-1",
                 "pyannote/segmentation-3.0"):
        try:
            api.model_info(repo, token=tok)
        except GatedRepoError:
            sys.exit(f"Lisensen for {repo} er ikke godtatt av «{who.get('name', '?')}»,"
                     " som er kontoen dette tokenet tilhører.\n"
                     f"  godta på https://huggingface.co/{repo}")
        except RepositoryNotFoundError:
            sys.exit(f"{repo} er ikke synlig for dette tokenet.\n"
                     "  feil konto, eller tokenet mangler read-tilgang.")
        except HfHubHTTPError as e:
            sys.exit(f"Hugging Face svarte med feil for {repo}: {e}")
        except Exception as e:
            sys.exit(f"Nådde ikke Hugging Face: {e}")
    print(f"access ok  ({who.get('name', '?')})")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selfcheck":
        _selfcheck()
    elif len(sys.argv) == 2 and sys.argv[1] == "--check-access":
        _check_access()
    else:
        try:
            main()
        except KeyboardInterrupt:
            # 128+signal, som et skall forventer. Ctrl-C før handleren er
            # installert gir _signum None, og det er SIGINT.
            sys.exit(143 if _signum == signal.SIGTERM else 130)
