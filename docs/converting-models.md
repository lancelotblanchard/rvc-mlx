# Tutorial: converting your RVC voices to MLX

This guide takes the voice models you trained (or downloaded) for the
[RVC WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) and turns them into files that
run natively on Apple silicon: in Python, in the [local web UI](../examples/web-server), in the
[iOS app](../examples/ios) and in the [JUCE / ARA plugin](../examples/juce-plugin). It takes about five minutes.

```
  RVC WebUI files                          rvc-mlx files                       used by
  ───────────────                          ─────────────                       ───────
  hubert_base.pt ─┐                     ┌─ hubert.safetensors   ┐
  rmvpe.pt ───────┴─ convert-base ──────┴─ rmvpe.safetensors    ├── every voice (shared)
                                                                │
  alice.pth ──────┐                                             │
  added_*.index ──┴─ convert-voice ──────── voices/alice.safetensors ── one file per voice
```

Conversion needs PyTorch once; running the converted files never does.

---

## 1. Install

```bash
git clone https://github.com/lancelotblanchard/rvc-mlx.git
cd rvc-mlx
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[convert]"      # rvc-mlx + torch + faiss-cpu (needed only for converting)
```

Python 3.10–3.12 on an Apple-silicon Mac is the reference setup.

## 2. Collect the files

| File | Where to find it |
| --- | --- |
| `hubert_base.pt` | RVC WebUI `assets/hubert/`, or [Hugging Face: lj1995/VoiceConversionWebUI](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/hubert_base.pt) |
| `rmvpe.pt` | RVC WebUI `assets/rmvpe/`, or [the same repo](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.pt) |
| `<voice>.pth` | RVC WebUI `assets/weights/` (the small *exported* model, typically 25–60 MB) |
| `added_*.index` | RVC WebUI `logs/<experiment>/` (optional, but improves timbre) |

> **`G_2333.pth` / `D_2333.pth` are training checkpoints, not voices.** Export them first in the WebUI
> (*ckpt processing → extract small model*), or use the `.pth` in `assets/weights/` that training writes
> automatically. The converter tells you if you give it the wrong one.

ContentVec exported for Hugging Face `transformers` (`pytorch_model.bin` / `model.safetensors`, e.g. the one Applio
uses) also works in place of `hubert_base.pt`.

## 3. Convert the shared models (once)

```bash
rvc-mlx convert-base --hubert hubert_base.pt --rmvpe rmvpe.pt -o models/
```

```
hubert -> models/hubert.safetensors
rmvpe  -> models/rmvpe.safetensors
```

Weights are stored in float16 by default (half the size, no audible difference). Add `--dtype float32` to keep full
precision.

## 4. Convert each voice

```bash
rvc-mlx convert-voice alice.pth \
    --index added_IVF1024_Flat_nprobe_1_alice_v2.index \
    --name "Alice" \
    -o models/voices/
```

```
Voice('Alice', v2, 40000 Hz, f0=True, index 48213x768) -> models/voices/alice.safetensors
```

Everything the runtime needs ends up in that one file: weights, the architecture (v1/v2, 32k/40k/48k, with or without
pitch), the speaker table and the retrieval index.

**Options**

| Flag | Use it when |
| --- | --- |
| `--index` | Always if you have one. Accepts the faiss `.index` or the `total_fea.npy` next to it. |
| `--max-index-vectors 20000` | Shrinks a large index with k-means: smaller files and faster retrieval for iPhone. 10–20k is plenty. |
| `--dtype float32` | You want bit-for-bit full precision. |
| `--name` | The display name shown in the apps (defaults to the file name). |

Converting a whole folder:

```bash
for pth in weights/*.pth; do
  name=$(basename "$pth" .pth)
  index=$(ls logs/"$name"/added_*.index 2>/dev/null | head -1)
  rvc-mlx convert-voice "$pth" ${index:+--index "$index"} -o models/voices/
done
```

## 5. Check the result

```bash
rvc-mlx info models/voices/alice.safetensors
rvc-mlx infer -m models/ -v models/voices/alice.safetensors my_voice.wav alice.wav --pitch 0
```

`--pitch` is in semitones: try `+12` for a low voice into a high one and `-12` the other way round. Other knobs
mirror the WebUI: `--index-rate` (0–1, default 0.75), `--protect` (0–0.5, default 0.33), `--rms-mix-rate` (0–1,
default 0.25).

## 6. Blend voices (optional)

Voices that share an architecture (same version and sample rate) can be blended by interpolating their weights,
the same thing as the WebUI's *ckpt merge*:

```bash
rvc-mlx blend models/voices/alice.safetensors:0.7 models/voices/bob.safetensors:0.3 \
    -o models/voices/alice_bob.safetensors --name "Alice × Bob"
```

The plugin does this live with a crossfader.

## 7. Try it in the browser

```bash
pip install -e ".[server]"
python examples/web-server/server.py --models models/
```

Open <http://127.0.0.1:8000>, record or drop a file, pick a voice, convert.

## 8. Use the files elsewhere

The `models/` folder is all the apps need:

- **iOS app**: AirDrop or Files-copy `hubert.safetensors`, `rmvpe.safetensors` and any voices to the phone, then tap
  *Import* in the app. Files are recognised by their contents, so names don't matter. See [examples/ios](../examples/ios).
- **Plugin**: point *Settings → Models folder* at `models/`. See [examples/juce-plugin](../examples/juce-plugin).
- **Python**:

  ```python
  from rvc_mlx import RVC, Voice, blend_voices

  rvc = RVC.from_pretrained("models/")
  alice = Voice.load("models/voices/alice.safetensors")
  audio, sr = rvc.convert_file("in.wav", "out.wav", alice, pitch=0, index_rate=0.75)

  # programmatic blending
  bob = Voice.load("models/voices/bob.safetensors")
  mix = blend_voices([alice, bob], [0.5, 0.5])
  ```

---

## Troubleshooting

**`doesn't look like an RVC voice export`**: you passed a training checkpoint (`G_*.pth`). See step 2.

**`Reading a faiss .index needs pip install faiss-cpu`**: install it (`pip install faiss-cpu`) or pass
`total_fea.npy` instead of the `.index`.

**`Index vectors have shape (N, 256), but a v2 voice needs (N, 768)`**: the index belongs to a different (v1)
experiment. Use the index from the same training run.

**Can I blend a v1 with a v2 voice, or 40k with 48k?** No. Blending averages weights, so the architectures must be
identical. `rvc-mlx blend` tells you which hyper-parameters differ.

**Models from forks (Applio etc.)** convert if they use the standard RVC v1/v2 HiFi-GAN-NSF decoder. Alternative
vocoders (RefineGAN, MRF-HiFi-GAN) are not supported.

**Voices without pitch guidance** (`f0 = 0`, "nono" models) are supported; the pitch control has no effect on them.

---

## Appendix: the file format

Every converted file is a standard [safetensors](https://github.com/huggingface/safetensors) file, so any runtime can
read it. The header metadata says what it is:

| key | value |
| --- | --- |
| `format` / `format_version` | `rvc-mlx` / `1` |
| `kind` | `voice`, `hubert` or `rmvpe` |
| `config` | JSON architecture description |
| `name`, `info`, `version`, `f0`, `sample_rate`, `index_size`, `merged_from` | voices only |

Tensors use MLX layouts: convolution kernels are channels-last (`[out, kernel, in]`), weight normalisation is already
folded into `*.weight`, and names otherwise match RVC's own state-dict keys. A voice's retrieval bank is the
`index.vectors` tensor (`[N, 256 | 768]`). `rmvpe.safetensors` also carries its mel filterbank (`mel_basis`), so the
Swift and C++ runtimes don't need librosa.
