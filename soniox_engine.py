#!/usr/bin/env python3
"""Soniox (sky) som transkriberings-motor — batch/async.

Alternativ til den lokale mlx-whisper + pyannote-stacken. Soniox er norsk-tunet
og krever ingen gated HF-modeller. Talere skilles på to måter:

  diarized:  én mikset fil  -> Soniox' egen sky-diarisering
  dual:      to mono-filer  -> kjent taler per fil, flettet på start_ms

Begge returnerer segmenter på samme form som resten av repoet bruker:
    {"start": s, "end": s, "speaker": str, "language": str, "text": str}
så `write_outputs()` i transcribe.py kan skrive .txt/.srt/.json uendret.

Soniox skiller ikke språk per taler slik mlx-motoren gjør (språk = norsk her);
`language`-feltet settes til language_hint for kompatibilitet med output-formatet.

Krever:
  - `~/.soniox/api-key` (eller env SONIOX_API_KEY)
  - nettverk mot api.soniox.com
"""
import json
import os
import time
import urllib.request

BASE = "https://api.soniox.com/v1"

# Egennavn Soniox ellers feilhører ("Oschlo" -> "Oslo"). Utvid ved behov.
DEFAULT_TERMS = [
    "Oschlo", "Fredrik Evjen Ekli", "Helen", "arealtilsyn", "strandsonetilsyn",
    "ulovlighetsoppfølging", "byggesak", "matrikkel", "strandsone",
    "dispensasjon", "flyfoto", "plan- og bygningsloven", "Statsforvalteren",
]


def _key():
    if os.environ.get("SONIOX_API_KEY"):
        return os.environ["SONIOX_API_KEY"].strip()
    path = os.path.expanduser("~/.soniox/api-key")
    if os.path.isfile(path):
        return open(path).read().strip()
    raise SystemExit(
        "Soniox-nøkkel mangler. Sett SONIOX_API_KEY eller legg den i ~/.soniox/api-key"
    )


def _req(method, path, data=None, headers=None, raw=False, key=None):
    h = {"Authorization": f"Bearer {key}"}
    if headers:
        h.update(headers)
    body = data if raw else (json.dumps(data).encode() if data is not None else None)
    req = urllib.request.Request(BASE + path, data=body, headers=h, method=method)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def _upload(audio, key):
    """Last opp en lydfil som multipart/form-data -> file_id."""
    boundary = "----maclocaltranscribe"
    fn = os.path.basename(audio)
    pre = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
           f"filename=\"{fn}\"\r\nContent-Type: audio/wav\r\n\r\n").encode()
    post = f"\r\n--{boundary}--\r\n".encode()
    body = pre + open(audio, "rb").read() + post
    d = _req("POST", "/files", data=body, raw=True, key=key,
             headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    return d["id"]


def _run_transcription(file_id, key, diarization, language, terms, poll=150):
    payload = {
        "model": "stt-async-v4", "file_id": file_id,
        "language_hints": [language],
        "enable_speaker_diarization": diarization,
        "context": {"terms": terms},
    }
    tid = _req("POST", "/transcriptions", data=payload, key=key)["id"]
    for _ in range(poll):
        st = _req("GET", f"/transcriptions/{tid}", key=key).get("status")
        if st == "completed":
            break
        if st == "error":
            raise RuntimeError("Soniox-transkribering feilet")
        time.sleep(4)
    else:
        raise RuntimeError("Soniox-transkribering tidsavbrutt")
    return _req("GET", f"/transcriptions/{tid}/transcript", key=key).get("tokens", [])


def _tokens_to_segments(tokens, speaker_of, language):
    """Grupper sammenhengende tokens med samme taler -> segmenter (sekunder)."""
    segs, cur, buf, start_ms, end_ms = [], None, "", 0, 0
    for t in tokens:
        text = t.get("text", "")
        if not text:
            continue
        spk = speaker_of(t)
        if cur is not None and spk != cur and buf.strip():
            segs.append({"start": start_ms / 1000, "end": end_ms / 1000,
                         "speaker": cur, "language": language, "text": buf.strip()})
            buf = ""
        if spk != cur or not buf:
            start_ms = t.get("start_ms", end_ms)
        cur = spk
        buf += text
        end_ms = t.get("end_ms", t.get("start_ms", start_ms))
    if buf.strip():
        segs.append({"start": start_ms / 1000, "end": end_ms / 1000,
                     "speaker": cur, "language": language, "text": buf.strip()})
    return segs


def transcribe_diarized(audio, language="no", terms=None):
    """Mikset fil -> Soniox' egen sky-diarisering -> segmenter."""
    key = _key()
    terms = terms or DEFAULT_TERMS
    print("  laster opp til Soniox …")
    fid = _upload(audio, key)
    print("  transkriberer (diarisering i sky) …")
    toks = _run_transcription(fid, key, diarization=True, language=language, terms=terms)
    segs = _tokens_to_segments(
        toks, lambda t: f"SPEAKER_{t.get('speaker', '0')}", language)
    print(f"  {len(toks)} tokens -> {len(segs)} segmenter")
    return segs


def transcribe_dual(channels, language="no", terms=None):
    """[(fil, taler-navn), …] -> hver fil transkriberes separat, flettet på start_ms.

    Kanalene må være tatt opp av samme klokke (delt nullpunkt) for riktig fletting.
    """
    key = _key()
    terms = terms or DEFAULT_TERMS
    tokens = []
    for audio, speaker in channels:
        if not (os.path.exists(audio) and os.path.getsize(audio) > 1000):
            print(f"  hopper over {audio} (mangler/tom)")
            continue
        print(f"  {speaker}: laster opp {os.path.basename(audio)} …")
        fid = _upload(audio, key)
        toks = _run_transcription(fid, key, diarization=False, language=language, terms=terms)
        for t in toks:
            t["_speaker"] = speaker
        tokens += toks
    tokens.sort(key=lambda t: t.get("start_ms", 0))
    segs = _tokens_to_segments(tokens, lambda t: t["_speaker"], language)
    print(f"  {len(tokens)} tokens -> {len(segs)} segmenter")
    return segs
