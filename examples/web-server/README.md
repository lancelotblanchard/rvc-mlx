# Local web UI

A single-page voice converter served from your Mac. Record in the browser (or drop a file), pick a voice, convert,
listen, download.

```bash
pip install -e ".[server]"            # from the repo root
python examples/web-server/server.py --models models/
# → http://127.0.0.1:8000
```

`models/` is the folder produced by the [conversion tutorial](../../docs/converting-models.md):

```
models/
├── hubert.safetensors
├── rmvpe.safetensors
└── voices/
    ├── alice.safetensors
    └── bob.safetensors
```

Voices are rescanned on every page load, so you can convert or blend new ones while the server is running.

**API** (handy for scripting):

```bash
curl -s localhost:8000/api/voices
curl -s -F audio=@me.wav -F voice=alice -F pitch=12 localhost:8000/api/convert -o out.wav
```

Form fields: `voice` (file stem in `voices/`), `pitch` (semitones), `index_rate`, `protect`, `rms_mix_rate`. The
response is a 16-bit WAV at the voice's sample rate; the `X-Elapsed` / `X-Duration` headers report timing.
The browser encodes everything to WAV before uploading; from scripts send WAV, FLAC or OGG.
