"""FLEURS-baseline: WER per språk og språk-ID (fri og begrenset til no/sv/en).

    .venv/bin/python bench/fleurs.py [--model REPO] [--limit N] [--langs nb_no,sv_se,en_us]

Lyden lastes ned én gang fra google/fleurs (CC-BY 4.0) til HF-cachen. Alt
annet er lokalt. Én JSONL-linje per klipp i work/fleurs/<modell>/, så en
avbrutt kjøring fortsetter der den slapp. Sammendraget skrives til slutt.

WER er mot FLEURS' normaliserte `transcription` (små bokstaver, uten
tegnsetting), og hypotesen normaliseres likt. Tall er ikke normalisert, så
«1970» mot «nittensytti» teller som feil — likt for alle modeller, og det er
relative tall vi er ute etter.
"""
import argparse, csv, json, os, re, sys, tarfile, time

import numpy as np, soundfile as sf
import mlx.core as mx
from huggingface_hub import hf_hub_download
import mlx_whisper
from mlx_whisper.audio import log_mel_spectrogram, pad_or_trim, N_FRAMES, N_SAMPLES
from mlx_whisper.transcribe import ModelHolder

WLANG = {"nb_no": "no", "sv_se": "sv", "en_us": "en"}
ALLOWED = ("no", "nn", "sv", "en")   # nn telles som no — samme som pipelinen burde gjøre

csv.field_size_limit(sys.maxsize)


def norm(t):
    return re.sub(r"[^\w\s']", " ", t.lower()).split()


def edits(ref, hyp):
    """Levenshtein på ordnivå. ponytail: O(n·m) uten jiwer — setningene er korte."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


assert edits("a b c".split(), "a x c d".split()) == 2  # ett bytte + ett innskudd


def fetch(lang):
    get = lambda f: hf_hub_download("google/fleurs", f, repo_type="dataset")
    tsv, tgz = get(f"data/{lang}/test.tsv"), get(f"data/{lang}/audio/test.tar.gz")
    audio_dir = tgz + ".d"
    if not os.path.isdir(audio_dir):
        with tarfile.open(tgz) as t:
            t.extractall(audio_dir + ".tmp", filter="data")
        os.rename(audio_dir + ".tmp", audio_dir)
    wavs = {os.path.basename(p): os.path.join(r, p)
            for r, _, fs in os.walk(audio_dir) for p in fs if p.endswith(".wav")}
    rows = [r for r in csv.reader(open(tsv), delimiter="\t", quoting=csv.QUOTE_NONE)]
    return [(r[1], r[3], wavs[r[1]]) for r in rows if r[1] in wavs]


def lang_probs(model, audio):
    mel = log_mel_spectrogram(audio, n_mels=model.dims.n_mels, padding=N_SAMPLES)
    _, probs = model.detect_language(pad_or_trim(mel, N_FRAMES, axis=-2).astype(mx.float16))
    return probs


def run(model_repo, lang, limit, out_dir):
    items = fetch(lang)[:limit] if limit else fetch(lang)
    path = os.path.join(out_dir, f"{lang}.jsonl")
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["file"] for l in open(path) if l.strip()}
    model = ModelHolder.get_model(model_repo, mx.float16)
    with open(path, "a") as f:
        for i, (name, ref, wav) in enumerate(items, 1):
            if name in done:
                continue
            audio, sr = sf.read(wav, dtype="float32")
            assert sr == 16000, (wav, sr)
            t = time.time()
            probs = lang_probs(model, audio)
            hyp = mlx_whisper.transcribe(audio, path_or_hf_repo=model_repo,
                                         language=WLANG[lang],
                                         condition_on_previous_text=False)["text"].strip()
            sec = time.time() - t
            free = max(probs, key=probs.get)
            restricted = max(ALLOWED, key=lambda k: probs.get(k, 0))
            f.write(json.dumps({"file": name, "ref": ref, "hyp": hyp,
                                "lid_free": free, "lid_restricted": restricted,
                                "p": {k: round(probs.get(k, 0), 4) for k in ALLOWED},
                                "dur": len(audio) / sr, "sec": sec},
                               ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0:
                print(f"  {lang} {i}/{len(items)}", flush=True)


def summarize(out_dir, langs):
    print(f"\n{'språk':6} {'n':>4} {'WER':>6} {'LID fri':>8} {'LID begr.':>9} {'→sv/no':>7} {'RTF':>6}")
    res = {}
    for lang in langs:
        rows = [json.loads(l) for l in open(os.path.join(out_dir, f"{lang}.jsonl"))]
        e = sum(edits(norm(r["ref"]), norm(r["hyp"])) for r in rows)
        n = sum(len(norm(r["ref"])) for r in rows)
        want = WLANG[lang]
        ok = lambda k: (k in ("no", "nn")) if want == "no" else k == want
        has_lid = rows[0]["lid_free"] is not None   # parakeet har ingen språk-ID
        free = sum(ok(r["lid_free"]) for r in rows) / len(rows) if has_lid else float("nan")
        restr = sum(ok(r["lid_restricted"]) for r in rows) / len(rows) if has_lid else float("nan")
        other = {"no": ("sv",), "sv": ("no", "nn"), "en": ()}[want]
        conf = sum(r["lid_restricted"] in other for r in rows)
        rtf = sum(r["sec"] for r in rows) / sum(r["dur"] for r in rows)
        res[lang] = {"n": len(rows), "wer": e / n, "lid_free": free,
                     "lid_restricted": restr, "no_sv_confusions": conf, "rtf": rtf}
        print(f"{lang:6} {len(rows):4} {100*e/n:5.1f}% {100*free:7.1f}% {100*restr:8.1f}% {conf:7} {rtf:6.3f}")
    json.dump(res, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/whisper-large-v3-mlx")
    ap.add_argument("--langs", default="nb_no,sv_se,en_us")
    ap.add_argument("--limit", type=int, default=0, help="maks klipp per språk (0 = alle)")
    a = ap.parse_args()
    langs = a.langs.split(",")
    out = os.path.join(os.path.dirname(__file__), "..", "work", "fleurs", a.model.split("/")[-1])
    os.makedirs(out, exist_ok=True)
    for lang in langs:
        run(a.model, lang, a.limit, out)
    summarize(out, langs)
