"""FLEURS for Parakeet TDT v3 (parakeet-mlx), samme format og sammendrag som fleurs.py.

    <venv med parakeet-mlx>/bin/python bench/fleurs_parakeet.py [--langs sv_se,en_us]

Parakeet kan ikke låses til et språk og gir ingen språk-ID, så LID-kolonnene
står tomme. Ingen norsk i v3 — nb_no er derfor ikke standard.
"""
import argparse, json, os, sys, time

import soundfile as sf
from parakeet_mlx import from_pretrained

sys.path.insert(0, os.path.dirname(__file__))
from fleurs import fetch, summarize

MODEL = "mlx-community/parakeet-tdt-0.6b-v3"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="sv_se,en_us")
    langs = ap.parse_args().langs.split(",")
    out = os.path.join(os.path.dirname(__file__), "..", "work", "fleurs", MODEL.split("/")[-1])
    os.makedirs(out, exist_ok=True)
    model = from_pretrained(MODEL)
    for lang in langs:
        path = os.path.join(out, f"{lang}.jsonl")
        done = {json.loads(l)["file"] for l in open(path)} if os.path.exists(path) else set()
        with open(path, "a") as f:
            for name, ref, wav in fetch(lang):
                if name in done:
                    continue
                t = time.time()
                hyp = model.transcribe(wav).text.strip()
                f.write(json.dumps({"file": name, "ref": ref, "hyp": hyp,
                                    "lid_free": None, "lid_restricted": None,
                                    "dur": sf.info(wav).duration, "sec": time.time() - t},
                                   ensure_ascii=False) + "\n")
                f.flush()
    summarize(out, langs)
