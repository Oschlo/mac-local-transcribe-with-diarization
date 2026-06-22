# Plan: Lokal transkripsjon med diarization (norsk + svensk i samme samtale)

## Mål

Transkribere videofiler hvor flere talere snakker, og hvor språket veksler **mellom** talere (én taler holder seg til norsk, en annen til svensk). Alt skal kjøre lokalt på en MacBook Pro M1 Pro (32 GB RAM, macOS Tahoe 26.5.1). Ingen skytjenester under selve kjøringen. Eneste tillatte nettverkskall er engangsnedlasting av modellvekter.

## Kjernestrategi

Kjør jobben i to omganger og utnytt at hver taler bruker ett fast språk:

1. **Diarization** med pyannote (språkuavhengig) gir segmenter med taler-ID og tidsstempler.
2. **Språkgruppering**: bestem hvilket språk hver taler-ID bruker (én gang per fil).
3. **Transkribering** med mlx-whisper, hvor `language` settes eksplisitt per segment basert på talerens språk. Da slipper vi at Whisper bommer på språkdeteksjon eller "oversetter" mellom norsk og svensk.
4. **Fletting** av alt til ett kronologisk transkript med taler- og språkmerking.

Begrunnelse for verktøyvalg: WhisperX er bygget rundt CUDA/NVIDIA og faller tilbake til CPU på Apple Silicon (tregt). mlx-whisper bruker Apples MLX-rammeverk og kjører raskt på M1 Pro. Diarization gjøres med pyannote direkte, siden mlx-whisper ikke har det innebygd.

## Miljø og avhengigheter

- Python 3.11 eller nyere, i et eget virtuelt miljø (`venv` eller `uv`).
- `ffmpeg` installert via Homebrew (`brew install ffmpeg`).
- Python-pakker:
  - `mlx-whisper` (transkribering på Apple Silicon)
  - `pyannote.audio` (diarization)
  - `torch` og `torchaudio` (kreves av pyannote; CPU/MPS-bygg er greit)
  - `soundfile` og `numpy` (lydhåndtering)
- Hugging Face-token kreves for pyannote. Modellen `pyannote/speaker-diarization-3.1` krever at man aksepterer vilkårene på Hugging Face én gang og genererer et access token. Token settes som miljøvariabel `HF_TOKEN`.
  - **Avklar med kunden** at engangsnedlasting av modellvekter fra Hugging Face er greit. Etter nedlasting kan det kjøres helt offline.
- Whisper-modell: `mlx-community/whisper-large-v3-mlx`. `large-v3` gir klart best kvalitet på norsk og svensk. 32 GB RAM er rikelig.

## Steg for steg

### Steg 0: Inspisere og forberede lyd

Videoene er MP4. Whisper og pyannote bruker bare lydsporet, så bildet er irrelevant. Først inspiser lydsporet, deretter trekk ut lyd.

**0a. Inspiser lydsporene først:**

```bash
ffprobe input.mp4
```

Sjekk hvor mange lydspor (streams) videoen har og hvor mange kanaler hvert spor har. To tilfeller:

- **Vanlig opptak (ett spor, alle talere mikset sammen):** Gå videre til mono-uttrekk under (0b). Dette er standardveien, og diarization i Steg 1 gjør jobben med å skille talerne.
- **Flere lydspor eller kanaler per taler (f.eks. separat mikrofon per person, eller én taler i venstre kanal og én i høyre):** Da har du en enklere og mer presis vei til diarization. Splitt kanalene/sporene til separate filer, transkriber hver for seg med talerens språk, og merk taler ut fra hvilken kanal lyden kom fra. Dette omgår pyannote helt og gir nær perfekt taler-skille. Hvis dette tilfellet oppdages, vurder denne ruten i stedet for pyannote-pipelinen.

**0b. Trekk ut lyd til 16 kHz mono WAV (formatet pyannote og Whisper foretrekker):**

```bash
ffmpeg -i input.mp4 -ar 16000 -ac 1 -c:a pcm_s16le audio.wav
```

(For kanal-splitting i flerspors-tilfellet, bruk i stedet f.eks. `ffmpeg -i input.mp4 -map_channel 0.0.0 taler_venstre.wav -map_channel 0.0.1 taler_hoyre.wav`, tilpasset faktisk kanaloppsett fra ffprobe.)

### Steg 1: Diarization

Kjør pyannote `speaker-diarization-3.1` på WAV-fila. Resultatet er en liste med segmenter: `(start, slutt, taler_id)`. Lagre dette som mellomresultat (f.eks. JSON), så man kan inspisere og eventuelt kjøre transkriberingen på nytt uten å diarisere igjen.

Praktiske hensyn:
- Hvis antall talere er kjent på forhånd, kan det sendes inn som hint (`num_speakers` eller `min/max_speakers`) for bedre resultat.
- Slå sammen sammenhengende segmenter fra samme taler med svært korte mellomrom, for å unngå unødvendig oppstykking.

### Steg 2: Språkgruppering per taler

Bestem hvilket språk hver taler-ID bruker. To mulige metoder, velg én:

- **Halvautomatisk (anbefalt for kvalitet):** Plukk det lengste segmentet per taler, transkriber det med mlx-whisper i autodetekter-modus, og les av detektert språk. Bygg en oppslagstabell `{taler_id: språk}`. La det være mulig å overstyre manuelt, siden norsk og svensk kan forveksles på korte klipp.
- **Manuell:** Skriv ut et par sekunder lyd per taler, la bruker lytte og fylle inn språk i en enkel mapping (f.eks. `{"SPEAKER_00": "no", "SPEAKER_01": "sv"}`).

Mappingen bør lagres slik at den kan redigeres og gjenbrukes.

### Steg 3: Segmentvis transkribering

For hvert diarization-segment:
- Klipp ut lydbiten med ffmpeg (eller i minnet med soundfile basert på tidsstempel).
- Kjør mlx-whisper på biten med `language` satt eksplisitt til talerens språk fra mappingen i steg 2.
- Bruk modell `mlx-community/whisper-large-v3-mlx`.

Merk: Veldig korte segmenter (under ~1 sekund) kan gi dårlig resultat. Vurder å slå sammen nabosegmenter fra samme taler før transkribering, eller å gi litt overlapp/kontekst i kantene.

### Steg 4: Fletting og output

Sorter alle transkriberte segmenter kronologisk på starttidspunkt og sett sammen til ett transkript. Skriv ut i (minst) disse formatene:

- **Lesbar tekst** med taler og tidsstempel per linje, f.eks.:
  ```
  [00:00:04] SPEAKER_00 (no): God morgen, skal vi sette i gang?
  [00:00:09] SPEAKER_01 (sv): Ja, absolut. Jag har förberett underlaget.
  ```
- **SRT/VTT** undertekster, hvis det trengs for video.
- **JSON** med full struktur (start, slutt, taler, språk, tekst) for videre bearbeiding.

## Filstruktur (forslag)

```
prosjekt/
  input/                  # videofiler
  work/                   # uttrukket lyd, segmentklipp, mellomresultater
    audio.wav
    diarization.json      # steg 1
    speaker_languages.json# steg 2 (redigerbar)
  output/
    transcript.txt
    transcript.srt
    transcript.json
  transcribe.py           # hovedskript
  requirements.txt
```

## Skript som bør lages

1. `extract_audio.py` (eller bare ffmpeg-kommandoer): inspiser med ffprobe, deretter video til 16 kHz mono WAV. Bør håndtere både standard mono-uttrekk og flerspors/kanal-splitting hvis det oppdages.
2. `diarize.py`: pyannote-diarization til `diarization.json`.
3. `detect_speaker_languages.py`: bygg `speaker_languages.json` (steg 2), med mulighet for manuell overstyring.
4. `transcribe_segments.py`: segmentvis mlx-whisper med språk per taler.
5. `merge_output.py`: flett til txt/srt/json.

Eventuelt samlet i ett `transcribe.py` med delkommandoer, slik at man kan kjøre stegene hver for seg og gjenbruke mellomresultater.

## Ting å være obs på

- **Norsk/svensk-forvirring** er kjernerisikoen. Den løses i stor grad av at språk låses per taler, men dobbeltsjekk språkgrupperingen i steg 2 manuelt før full kjøring.
- **Whisper kan oversette** mellom nært beslektede språk hvis språk ikke er låst. Sørg for at `language` alltid settes eksplisitt i steg 3, aldri autodetekter.
- **Overlappende tale** (folk snakker i munnen på hverandre) håndteres begrenset av pyannote. Forvent litt rot der.
- **Korte segmenter** gir svakere transkripsjon; vurder sammenslåing.
- **Regn med manuell korrektur** til slutt uansett. Målet er å minimere den, ikke eliminere den.
- **Offline-kjøring**: Etter at modellvektene er lastet ned første gang, sett `HF_HUB_OFFLINE=1` for å garantere at ingenting går mot nett under produksjonskjøring. Greit å nevne overfor kunden.

## Test før full kjøring

Kjør hele pipelinen på en kort klippet testbit (1-2 minutter med både norsk og svensk taler) før de fulle filene. Verifiser at:
- Diarization skiller talerne riktig.
- Språk er korrekt mappet per taler.
- Transkripsjonen ikke har oversatt mellom språkene.
