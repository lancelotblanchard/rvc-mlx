from typing import Optional, List, Tuple, Union

import librosa
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from rvc_mlx.stft import stft
from rvc_mlx.utils import pad_constant
from rvc_mlx.windows import hann


class MelSpectrogram:
    """
    Mel spectrogram extractor matching the RVC reference implementation. Computes the log-mel spectrogram of an audio
    signal, with optional pitch-shifting (`keyshift`) and time-stretching (`speed`).

    The mel filterbank is precomputed using `librosa.filters.mel(..., htk=True)` to remain bit-exact with the original
    RVC reference implementation, which uses the same call. Subsequent operations (STFT, magnitude, mel projection, log)
    run on MLX arrays. The STFT is configured with `pad_mode="reflect"` and `onesided=True` to match
    `torch.stft`'s defaults for real input as used by RVC.
    """

    def __init__(
        self,
        is_half: bool,
        n_mel_channels: int,
        sampling_rate: int,
        win_length: int,
        hop_length: int,
        n_fft: Optional[int] = None,
        mel_fmin: float = 0,
        mel_fmax: Optional[float] = None,
        clamp: float = 1e-5,
    ):
        n_fft = win_length if n_fft is None else n_fft
        mel_basis = librosa.filters.mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
            htk=True,
        ).astype(np.float32)
        self.mel_basis = mx.array(mel_basis)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp
        self.is_half = is_half
        self.hann_window: dict = {}

    def __call__(
        self,
        audio: mx.array,
        keyshift: int = 0,
        speed: int = 1,
        center: bool = True,
    ) -> mx.array:
        factor = 2 ** (keyshift / 12)
        n_fft_new = int(np.round(self.n_fft * factor))
        win_length_new = int(np.round(self.win_length * factor))
        hop_length_new = int(np.round(self.hop_length * speed))

        keyshift_key = str(keyshift)
        if keyshift_key not in self.hann_window:
            self.hann_window[keyshift_key] = hann(win_length_new, sym=False)

        fft = stft(
            audio,
            n_fft=n_fft_new,
            hop_length=hop_length_new,
            win_length=win_length_new,
            window=self.hann_window[keyshift_key],
            center=center,
            pad_mode="reflect",
            onesided=True,
        )
        magnitude = mx.sqrt(fft.real**2 + fft.imag**2)

        if keyshift != 0:
            target_size = self.n_fft // 2 + 1
            resize = magnitude.shape[-2]
            if resize < target_size:
                # Pad the frequency axis (second-to-last) on the right with zeros so it matches `target_size`.
                magnitude = pad_constant(magnitude, (0, 0, 0, target_size - resize), value=0.0)
            magnitude = magnitude[..., :target_size, :] * self.win_length / win_length_new

        mel_output = mx.matmul(self.mel_basis, magnitude)
        if self.is_half:
            mel_output = mel_output.astype(mx.float16)
        log_mel_spec = mx.log(mx.clip(mel_output, a_min=self.clamp, a_max=None))
        return log_mel_spec


class ConvBlockRes(nn.Module):
    """
    Residual convolutional block used throughout RVC's RMVPE U-Net. Two 3x3 conv + BN + ReLU stages followed by an
    additive shortcut. If `in_channels != out_channels`, the shortcut is a 1x1 conv that matches dimensions; otherwise
    it is the identity.

    Note: MLX uses channels-last convention for 2D convolutions. Callers should provide input in shape
    `(B, H, W, C)` rather than PyTorch's `(B, C, H, W)`. The internal structure mirrors the PyTorch reference so weights
    can be copied with a simple per-attribute transposition (see tests).
    """

    def __init__(self, in_channels: int, out_channels: int, momentum: float = 0.01):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self._has_shortcut = in_channels != out_channels
        if self._has_shortcut:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        residual = self.shortcut(x) if self._has_shortcut else x
        return self.conv(x) + residual


class ResEncoderBlock(nn.Module):
    """
    A stack of `n_blocks` `ConvBlockRes` modules followed by an optional `AvgPool2d`. When `kernel_size` is `None`, no
    pooling is applied and the block returns a single tensor (used by `Intermediate`). Otherwise it returns a
    `(skip, pooled)` tuple, where `skip` is the pre-pool feature map kept for the decoder and `pooled` is the
    downsampled output that continues through the encoder.

    Input/output are channels-last `(B, H, W, C)` in MLX convention.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Optional[tuple],
        n_blocks: int = 1,
        momentum: float = 0.01,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.conv = [ConvBlockRes(in_channels, out_channels, momentum)]
        for _ in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))
        self.kernel_size = kernel_size
        if self.kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def __call__(self, x: mx.array) -> Union[Tuple[mx.array, mx.array], mx.array]:
        for block in self.conv:
            x = block(x)
        if self.kernel_size is not None:
            return x, self.pool(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        in_size: int,
        n_encoders: int,
        kernel_size: Optional[tuple],
        n_blocks: int,
        out_channels: int = 16,
        momentum: float = 0.01,
    ):
        super().__init__()
        self.n_encoders = n_encoders
        self.bn = nn.BatchNorm(in_channels, momentum=momentum)
        self.layers = []
        self.latent_channels = []
        for i in range(self.n_encoders):
            self.layers.append(
                ResEncoderBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    n_blocks=n_blocks,
                    momentum=momentum,
                )
            )
            self.latent_channels.append([out_channels, in_size])
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.out_size = in_size
        self.out_channel = out_channels

    def __call__(self, x: mx.array) -> Tuple[mx.array, List[mx.array]]:
        concat_tensors: List[mx.array] = []
        x = self.bn(x)
        for i, layer in enumerate(self.layers):
            t, x = layer(x)
            concat_tensors.append(t)
        return x, concat_tensors


class Intermediate(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_inters: int,
        n_block: int,
        momentum: float = 0.01,
    ):
        super().__init__()
        self.n_inters = n_inters
        self.layers = [ResEncoderBlock(in_channels, out_channels, None, n_block, momentum)]
        for i in range(self.n_inters - 1):
            self.layers.append(ResEncoderBlock(out_channels, out_channels, None, n_block, momentum))

    def __call__(self, x: mx.array) -> mx.array:
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x


class ResDecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: Union[int, Tuple[int, int]],
        n_blocks: int = 1,
        momentum: float = 0.01,
    ):
        super().__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.n_blocks = n_blocks
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
                output_padding=out_padding,
                bias=False,
            ),
            nn.BatchNorm(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.conv2 = [ConvBlockRes(out_channels * 2, out_channels, momentum)]
        for i in range(n_blocks - 1):
            self.conv2.append(ConvBlockRes(out_channels, out_channels, momentum))

    def __call__(self, x: mx.array, concat_tensor: mx.array) -> mx.array:
        x = self.conv1(x)
        # MLX uses channels-last (B, H, W, C), so concatenate along the last axis (channels). The PyTorch reference
        # uses `dim=1`, which is the channel dim in PyTorch's channels-first convention.
        x = mx.concatenate((x, concat_tensor), axis=-1)
        for i, conv2 in enumerate(self.conv2):
            x = conv2(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_decoders: int,
        stride: Union[int, Tuple[int, int]],
        n_blocks: int,
        momentum: float = 0.01,
    ):
        super().__init__()
        self.layers = []
        self.n_decoders = n_decoders
        for i in range(self.n_decoders):
            out_channels = in_channels // 2
            self.layers.append(
                ResDecoderBlock(in_channels, out_channels, stride, n_blocks, momentum)
            )
            in_channels = out_channels

    def __call__(self, x: mx.array, concat_tensors: List[mx.array]) -> mx.array:
        for i, layer in enumerate(self.layers):
            x = layer(x, concat_tensors[-1 - i])
        return x


class DeepUnet(nn.Module):
    def __init__(
        self,
        kernel_size: tuple,
        n_blocks: int,
        en_de_layers: int = 5,
        inter_layers: int = 4,
        in_channels: int = 1,
        en_out_channels: int = 16,
    ):
        super().__init__()
        self.encoder = Encoder(
            in_channels, 128, en_de_layers, kernel_size, n_blocks, en_out_channels
        )
        self.intermediate = Intermediate(
            self.encoder.out_channel // 2, self.encoder.out_channel, inter_layers, n_blocks
        )
        self.decoder = Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

    def __call__(self, x: mx.array) -> mx.array:
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        x = self.decoder(x, concat_tensors)
        return x


class BiGRU(nn.Module):
    """
    Bidirectional, multi-layer GRU built from MLX's single-direction, single-layer `nn.GRU` primitives. Mirrors
    `torch.nn.GRU(input_features, hidden_features, num_layers=num_layers, batch_first=True, bidirectional=True)`.

    For each layer, the input is fed through a forward GRU and a backward GRU (which runs over the reversed sequence,
    then the output is re-reversed). The two outputs are concatenated along the feature axis, and the next layer takes
    that as input. The initial hidden state is explicit zeros so that the first time-step's bias contribution matches
    PyTorch's behavior (MLX `nn.GRU` skips the recurrent bias terms when `hidden is None`).
    """

    def __init__(self, input_features: int, hidden_features: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_features = hidden_features
        self.forward_grus: List[nn.GRU] = []
        self.backward_grus: List[nn.GRU] = []
        for i in range(num_layers):
            in_feat = input_features if i == 0 else 2 * hidden_features
            self.forward_grus.append(nn.GRU(in_feat, hidden_features))
            self.backward_grus.append(nn.GRU(in_feat, hidden_features))

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, D). Returns (B, T, 2 * hidden_features).
        # MLX 0.30 does not have `mx.flip`; we reverse the time axis (-2) with a negative-step slice instead.
        batch = x.shape[0]
        h0 = mx.zeros((batch, self.hidden_features))
        for fgru, bgru in zip(self.forward_grus, self.backward_grus):
            f_out = fgru(x, hidden=h0)
            b_out = bgru(x[:, ::-1, :], hidden=h0)
            b_out = b_out[:, ::-1, :]
            x = mx.concatenate([f_out, b_out], axis=-1)
        return x


class E2E(nn.Module):
    """
    End-to-end RMVPE pitch network: a `DeepUnet` followed by a 3x3 Conv, then a `BiGRU` + `Linear` + sigmoid head that
    projects each time step to 360 cents bins.

    The PyTorch reference takes `mel` in shape (B, n_mel, T) (channels-first) and rearranges via
    `mel.transpose(-1, -2).unsqueeze(1)` to (B, 1, T, n_mel). Our `__call__` accepts the same (B, n_mel, T) shape and
    does the equivalent rearrangement to MLX channels-last (B, T, n_mel, 1) internally.
    """

    def __init__(
        self,
        n_blocks: int,
        n_gru: int,
        kernel_size: tuple,
        en_de_layers: int = 5,
        inter_layers: int = 4,
        in_channels: int = 1,
        en_out_channels: int = 16,
    ):
        super().__init__()
        if not n_gru:
            raise NotImplementedError(
                "E2E without GRU layers is not supported (n_gru must be > 0)."
            )
        self.unet = DeepUnet(
            kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, en_out_channels
        )
        # The reference uses Conv2d(en_out_channels, 3, kernel_size=(3, 3), padding=(1, 1)).
        self.cnn = nn.Conv2d(en_out_channels, 3, kernel_size=3, padding=1)
        # BiGRU input is 3 * 128 = 384 because the U-Net preserves the in_size of 128 along the mel axis.
        self.gru = BiGRU(3 * 128, 256, n_gru)
        self.linear = nn.Linear(512, 360)
        # Dropout(0.25) and Sigmoid are part of the reference's `fc` Sequential. Dropout is a no-op in eval mode and
        # MLX's Sigmoid is just `mx.sigmoid`; we apply both explicitly so the eval-time output matches PyTorch exactly.

    def __call__(self, mel: mx.array) -> mx.array:
        # mel: (B, n_mel, T) -> (B, T, n_mel, 1) for MLX channels-last conv.
        x = mx.expand_dims(mx.transpose(mel, (0, 2, 1)), -1)
        x = self.unet(x)  # (B, T, n_mel, en_out_channels)
        x = self.cnn(x)  # (B, T, n_mel, 3)
        # PyTorch path is `x.transpose(1, 2).flatten(-2)` on (B, 3, T, n_mel) -> (B, T, 3, n_mel) -> (B, T, 3 * n_mel).
        # Mirror that ordering by permuting channels-last (B, T, n_mel, 3) -> (B, T, 3, n_mel) before flattening.
        x = mx.transpose(x, (0, 1, 3, 2))  # (B, T, 3, n_mel)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # (B, T, 3 * n_mel)
        x = self.gru(x)
        x = self.linear(x)
        x = mx.sigmoid(x)
        return x


# Hyperparameters of the released RVC `rmvpe.pt` checkpoint. Pinned here (and mirrored in `rvc_mlx._torch_ref` for
# the conversion side) so `RMVPE.from_pretrained` can build the right shape without the caller specifying them.
_DEFAULT_E2E_CONFIG = dict(
    n_blocks=4,
    n_gru=1,
    kernel_size=(2, 2),
    en_de_layers=5,
    inter_layers=4,
    in_channels=1,
    en_out_channels=16,
)


class RMVPE:
    """
    Orchestration wrapper around `MelSpectrogram` and the `E2E` pitch network. Mirrors the inference path of the RVC
    reference `RMVPE` class: extract a 128-bin log-mel spectrogram from raw audio, pass it through the network to get
    a 360-bin frame-level salience, then decode the salience to fundamental-frequency (f0) estimates in Hz.

    The reference loads its E2E weights from a PyTorch `.pt` checkpoint. Here we take the `E2E` instance directly via
    the constructor so the orchestration code stays decoupled from checkpoint loading. A separate helper can copy
    weights from a torch E2E into an MLX E2E using the paired-module bridge in tests.
    """

    # The 360 output bins span pitches starting at ~32.7 Hz (C1) in 20-cent increments. The reference pads four bins on
    # each side so the local-average-cents weighted mean has 9 neighbours to look at without bounds checks.
    _CENTS_OFFSET = 1997.3794084376191
    _CENTS_STEP = 20
    _NUM_BINS = 360
    _PAD = 4

    @classmethod
    def from_pretrained(cls, path: str, is_half: bool = False) -> "RMVPE":
        """
        Build an `RMVPE` with weights loaded from disk.

        Accepts either an MLX-native `.safetensors` file or the original RVC `.pt` checkpoint. For `.pt`, the file is
        converted on first use to a sibling `.safetensors` (via `rvc_mlx.convert`); subsequent loads skip the
        conversion. Torch is only imported when a `.pt` actually needs converting.
        """
        # Late import to avoid a circular import (convert.py imports rmvpe.E2E for the destination model).
        from rvc_mlx.convert import ensure_safetensors

        safetensors_path = ensure_safetensors(path, is_half=is_half)
        model = E2E(**_DEFAULT_E2E_CONFIG)
        model.load_weights(safetensors_path)
        # Inference-only: pin BatchNorm to use the running stats baked into the checkpoint. The train/eval flag is
        # not part of the safetensors, so a freshly-built MLX module defaults to train mode after `load_weights`.
        model.eval()
        return cls(model=model, is_half=is_half)

    def __init__(self, model: E2E, is_half: bool = False):
        self.is_half = is_half
        self.mel_extractor = MelSpectrogram(
            is_half=is_half,
            n_mel_channels=128,
            sampling_rate=16000,
            win_length=1024,
            hop_length=160,
            n_fft=None,
            mel_fmin=30,
            mel_fmax=8000,
        )
        self.model = model
        cents_mapping = self._CENTS_STEP * np.arange(self._NUM_BINS) + self._CENTS_OFFSET
        # Pad on both sides so windowed neighbours around bin 0 / NUM_BINS-1 stay in range without index clamping.
        self.cents_mapping = np.pad(
            cents_mapping, (self._PAD, self._PAD)
        )  # length: NUM_BINS + 2 * PAD = 368

    def mel2hidden(self, mel: mx.array) -> mx.array:
        """
        Run a log-mel spectrogram through the E2E network. The reference pads the time axis to a multiple of 32 (the
        encoder's spatial reduction factor is 2 ** 5 = 32) so the U-Net can downsample cleanly, then crops the model
        output back to the original number of frames.
        """
        n_frames = mel.shape[-1]
        n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
        if n_pad > 0:
            # Pad only the time (last) axis on the right. Constant (zero) padding matches the reference.
            mel = pad_constant(mel, (0, n_pad), value=0.0)
        if self.is_half:
            mel = mel.astype(mx.float16)
        hidden = self.model(mel)
        return hidden[..., :n_frames, :]

    def to_local_average_cents(self, salience: mx.array, thred: float = 0.05) -> np.ndarray:
        """
        Convert a frame-level salience map of shape (..., T, 360) into per-frame cents estimates using a 9-bin
        local weighted mean around the argmax bin. Frames whose peak salience is below `thred` are zeroed out.
        Mirrors the reference's `to_local_average_cents` and runs on NumPy because the indexing patterns are easier to
        express there and the cost is negligible compared to the network forward.
        """
        salience_np = np.array(salience)
        center = np.argmax(salience_np, axis=-1)  # (..., T)
        # Pad along the last dim so center +/- 4 stays in bounds.
        padded = np.pad(salience_np, [(0, 0)] * (salience_np.ndim - 1) + [(self._PAD, self._PAD)])
        # Build a windowed view of length 9 (= 2*PAD + 1) around each center.
        window_size = 2 * self._PAD + 1
        offsets = np.arange(window_size)  # 0..8
        # `center + offsets` (broadcasting) gives indices into the padded salience for each frame.
        idx = center[..., None] + offsets  # (..., T, 9)
        gathered = np.take_along_axis(padded, idx, axis=-1)  # (..., T, 9)
        # `cents_mapping` is also length 368; index it with the same window so the cents values align.
        cents_window = self.cents_mapping[idx]  # (..., T, 9)
        product_sum = np.sum(gathered * cents_window, axis=-1)
        weight_sum = np.sum(gathered, axis=-1)
        # Avoid division by zero; downstream we still threshold on max salience so degenerate frames get zeroed.
        weight_sum = np.where(weight_sum == 0, 1.0, weight_sum)
        devided = product_sum / weight_sum
        # Threshold: any frame whose max salience is below `thred` is treated as unvoiced (cents = 0).
        maxx = np.max(salience_np, axis=-1)
        devided = np.where(maxx <= thred, 0.0, devided)
        return devided

    def decode(self, hidden: mx.array, thred: float = 0.03) -> np.ndarray:
        """Convert a (..., T, 360) salience map to f0 in Hz; zero entries indicate unvoiced frames."""
        cents_pred = self.to_local_average_cents(hidden, thred=thred)
        f0 = 10 * (2 ** (cents_pred / 1200))
        # Where cents_pred == 0 the formula gives f0 == 10. The reference uses this sentinel to mark unvoiced frames
        # and zeroes them out.
        f0 = np.where(f0 == 10, 0.0, f0)
        return f0

    def infer_from_audio(self, audio: mx.array, thred: float = 0.03) -> np.ndarray:
        """
        Full RMVPE inference: raw audio in -> per-frame f0 (Hz) out. Matches the reference contract: `audio` is a 1D
        array of samples at 16 kHz, and the returned f0 is 1D of shape (T,). Use `mel2hidden` + `decode` directly if
        you need batched inference.
        """
        if audio.ndim != 1:
            raise ValueError(f"Expected a 1D audio array, got shape {audio.shape}")
        mel = self.mel_extractor(mx.expand_dims(audio, 0))
        hidden = self.mel2hidden(mel)
        return self.decode(hidden, thred=thred)[0]
