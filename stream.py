#!/usr/bin/env python3
"""Live transkripsjon av en Mac-samtale via Soniox realtime (standalone).

To kilder, hver sin Soniox-sesjon -> talermerking uten diarisering:
  - system-tap (helen_systemtap --stream)  = Motpart   (48 kHz s16le)
  - mikrofon   (ffmpeg avfoundation)       = Meg       (16 kHz s16le)

Tokens skrives live til stdout. Ved stopp (Ctrl-C) flettes et transkript til fil.
Mac-spesifikt: krever den medfølgende helen_systemtap-binæren (fanger motpartens
lyd via CoreAudio) og en mikrofon-enhet som ffmpeg/avfoundation ser.

Bruk:
  python stream.py [out.md]        # kjører til Ctrl-C, default /tmp/stream_<pid>.md

Miljø:
  SONIOX_API_KEY            (eller ~/.soniox/api-key)
  HELEN_SYSTEMTAP           sti til systemtap-binæren (default: ./helen_systemtap)
  STREAM_MIC_DEVICE         avfoundation-mic-navn (default: "MacBook Pro Microphone")
"""
import os
import signal
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SYSTEMTAP = os.environ.get("HELEN_SYSTEMTAP", os.path.join(HERE, "helen_systemtap"))
MIC_DEV = os.environ.get("STREAM_MIC_DEVICE", "MacBook Pro Microphone")

_KEY = os.path.expanduser("~/.soniox/api-key")
if os.path.isfile(_KEY) and not os.environ.get("SONIOX_API_KEY"):
    os.environ["SONIOX_API_KEY"] = open(_KEY).read().strip()

try:
    from soniox import SonioxClient
    from soniox.types import RealtimeSTTConfig
except ImportError:
    sys.exit("soniox-SDK mangler. Installer med: pip install soniox")

DEFAULT_TERMS = [
    "Oschlo", "Fredrik Evjen Ekli", "Helen", "arealtilsyn", "strandsonetilsyn",
    "byggesak", "ulovlighetsoppfølging", "matrikkel", "strandsone",
    "dispensasjon", "flyfoto", "plan- og bygningsloven", "Statsforvalteren",
]

_lines = []                  # (label, text) i ankomst-rekkefølge (~kronologisk)
_lock = threading.Lock()
_procs = []


def _cfg(rate, terms):
    return RealtimeSTTConfig(
        model="stt-rt-v4", audio_format="pcm_s16le", sample_rate=rate,
        num_channels=1, language_hints=["no"], enable_speaker_diarization=False,
        enable_endpoint_detection=True, context={"terms": terms},
    )


def worker(label, stdout, rate, terms):
    try:
        client = SonioxClient()
        with client.realtime.stt.connect(config=_cfg(rate, terms)) as session:
            def sender():
                while True:
                    chunk = stdout.read(rate // 5 * 2)  # ~100 ms
                    if not chunk:
                        try:
                            session.finish()
                        except Exception:
                            pass
                        return
                    try:
                        session.send_byte_chunk(chunk)
                    except Exception:
                        return
            threading.Thread(target=sender, daemon=True).start()

            buf = []

            def flush():
                txt = "".join(buf).strip()
                buf.clear()
                if not txt:
                    return
                with _lock:
                    _lines.append((label, txt))
                print(f"{label}: {txt}", flush=True)

            for event in session.receive_events():
                if event.error_code:
                    print(f"[{label}] Soniox {event.error_code}: {event.error_message}",
                          file=sys.stderr)
                    break
                for tok in event.tokens:
                    if not tok.is_final:
                        continue
                    if tok.text in ("<end>", "<fin>"):
                        flush()
                        continue
                    buf.append(tok.text)
                    if tok.text.strip().endswith((".", "?", "!")) and len("".join(buf)) > 20:
                        flush()
                if event.finished:
                    break
            flush()
    except Exception as e:
        print(f"[{label}] worker-feil: {e}", file=sys.stderr)


def finalize(out):
    for p in _procs:
        try:
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
        except Exception:
            pass
    time.sleep(1.0)
    with _lock:
        lines = list(_lines)
    if not lines:
        print("ingen tale fanget", file=sys.stderr)
        return
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    open(out, "w", encoding="utf-8").write(
        "\n\n".join(f"**{lbl}:** {txt}" for lbl, txt in lines) + "\n")
    print(f"\ntranskript: {out} ({len(lines)} segmenter)", file=sys.stderr)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else f"/tmp/stream_{os.getpid()}.md"
    if not os.path.exists(SYSTEMTAP):
        sys.exit(f"mangler systemtap-binær: {SYSTEMTAP} (sett HELEN_SYSTEMTAP)")

    tap = subprocess.Popen([SYSTEMTAP, "--stream"], stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL)
    mic = subprocess.Popen(["ffmpeg", "-f", "avfoundation", "-i", f":{MIC_DEV}",
                            "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    _procs.extend([tap, mic])

    stopping = {"v": False}

    def on_sig(*_):
        if stopping["v"]:
            return
        stopping["v"] = True
        finalize(out)
        os._exit(0)

    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)

    t1 = threading.Thread(target=worker, args=("Motpart", tap.stdout, 48000, DEFAULT_TERMS), daemon=True)
    t2 = threading.Thread(target=worker, args=("Meg", mic.stdout, 16000, DEFAULT_TERMS), daemon=True)
    t1.start()
    t2.start()
    print(f"streamer (Ctrl-C for å stoppe) -> {out}", file=sys.stderr)
    signal.pause()


if __name__ == "__main__":
    main()
