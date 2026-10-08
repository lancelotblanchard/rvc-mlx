# RVC Pocket: iOS app

Record (or import) audio on your iPhone and convert it to any of your RVC voices, fully on-device with MLX.

- One screen: **record → pick a voice → convert → play / share**.
- Pitch slider with *Deeper / Same / Higher* presets. *Advanced* exposes Index and Protect.
- Voices are the same `.safetensors` files the CLI, web UI and plug-in use. Import them with **AirDrop**, the
  **Files** app ("On My iPhone › RVC Pocket", or drag them in from a Mac with Finder) or the in-app **Import**
  button. Files are recognised by their contents, so names don't matter.

```
examples/ios/
├── project.yml      XcodeGen spec for the app
├── App/             SwiftUI app (RVC Pocket)
└── RVCKit/          Swift package: the RVC engine on mlx-swift (usable in your own apps)
```

## Run it

Requirements: Xcode 15+, an iPhone or iPad with Apple silicon (A14 / M1 or newer, iOS 17+). MLX needs a real
device; the Simulator doesn't run Metal compute.

```bash
brew install xcodegen
cd examples/ios
xcodegen                      # creates RVCPocket.xcodeproj
open RVCPocket.xcodeproj      # pick your team under Signing & Capabilities, then Run on your device
```

Then put your models on the phone:

1. On your Mac, convert the base models and your voices (see the [tutorial](../../docs/converting-models.md)).
   For phones, shrink large indexes: `rvc-mlx convert-voice my.pth --index my.index --max-index-vectors 20000`.
2. AirDrop `hubert.safetensors`, `rmvpe.safetensors` and your voice files to the phone and choose *RVC Pocket*,
   or copy them into the app's folder in Finder / Files.
3. The setup card ticks off each piece. Record and convert.

Memory: the app loads HuBERT and RMVPE once (about 380 MB in half precision) plus the selected voice (about 55 MB),
using the increased-memory entitlement. The first conversion includes model loading; later ones start immediately.

## RVCKit

```swift
import RVCKit

let engine = try RVCEngine(hubertURL: hubert, rmvpeURL: rmvpe)           // .float16 by default
let voice = try Voice.load(voiceURL)
let out = try engine.convert(resample(samples, from: 48000, to: 16000), voice: voice,
                             options: ConvertOptions(pitch: 12))
// `out` is at voice.info.sampleRate

let blend = try Voice.blend([alice, bob], weights: [0.7, 0.3])          // weight interpolation
```

RVCKit is a line-for-line port of the C++ engine (`examples/juce-plugin/engine`), which is itself tested against
the Python implementation. To check RVCKit on a Mac (with `xcodebuild`: SwiftPM's command line can't compile MLX's
Metal kernels):

```bash
python examples/juce-plugin/engine/tests/make_golden.py /tmp/rvc-golden    # from the repo root
cd examples/ios/RVCKit
TEST_RUNNER_RVC_GOLDEN_DIR=/tmp/rvc-golden xcodebuild test -scheme RVCKit -destination 'platform=macOS'
```

The same check runs in CI (`.github/workflows/tests.yml`), together with an unsigned build of the app.
