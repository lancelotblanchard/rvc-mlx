"""
RVC voice synthesizers in MLX: `SynthesizerTrnMs{256,768}NSFsid` (v1 / v2 with pitch) and their `_nono` variants.

The model maps frame-level content features (HuBERT, 100 frames/s after 2x upsampling) plus a coarse pitch track to
audio: a relative-position transformer prior (`enc_p`), a reversed normalising flow (`flow`) conditioned on a speaker
embedding (`emb_g`), and an NSF-HiFiGAN vocoder (`dec`) driven by a sine excitation built from the fine f0 track.

Conventions: activations are channels-last `(B, T, C)`; masks are `(B, T, 1)`. Parameter names are exactly the keys
of a released RVC `.pth` (with weight-norm pairs folded into `.weight`), which keeps conversion a mechanical step.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

LRELU_SLOPE = 0.1


@dataclass
class SynthConfig:
    """Architecture hyper-parameters, i.e. the `config` list stored in RVC `.pth` files (by name, minus training-only
    entries)."""

    inter_channels: int = 192
    hidden_channels: int = 192
    filter_channels: int = 768
    n_heads: int = 2
    n_layers: int = 6
    kernel_size: int = 3
    resblock: str = "1"
    resblock_kernel_sizes: List[int] = None
    resblock_dilation_sizes: List[List[int]] = None
    upsample_rates: List[int] = None
    upsample_initial_channel: int = 512
    upsample_kernel_sizes: List[int] = None
    spk_embed_dim: int = 109
    gin_channels: int = 256
    sr: int = 40000
    version: str = "v2"
    f0: bool = True

    def __post_init__(self):
        if self.resblock_kernel_sizes is None:
            self.resblock_kernel_sizes = [3, 7, 11]
        if self.resblock_dilation_sizes is None:
            self.resblock_dilation_sizes = [[1, 3, 5]] * 3
        if self.upsample_rates is None:
            self.upsample_rates = [10, 10, 2, 2]
        if self.upsample_kernel_sizes is None:
            self.upsample_kernel_sizes = [16, 16, 4, 4]
        if self.version not in ("v1", "v2"):
            raise ValueError(f"Unknown RVC version {self.version!r}")

    @property
    def feature_dim(self) -> int:
        """Width of the HuBERT features this voice expects."""
        return 256 if self.version == "v1" else 768

    @property
    def hop_length(self) -> int:
        """Output samples generated per input frame."""
        return math.prod(self.upsample_rates)

    @classmethod
    def from_rvc(cls, config: list, version: str = "v1", f0: bool = True) -> "SynthConfig":
        """Build from the positional `config` list of an RVC `.pth` (spec_channels, segment_size, inter_channels, ...)."""
        (
            _spec_channels,
            _segment_size,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            _p_dropout,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            spk_embed_dim,
            gin_channels,
            sr,
        ) = config
        if isinstance(sr, str):
            sr = {"32k": 32000, "40k": 40000, "48k": 48000}[sr]
        return cls(
            inter_channels=int(inter_channels),
            hidden_channels=int(hidden_channels),
            filter_channels=int(filter_channels),
            n_heads=int(n_heads),
            n_layers=int(n_layers),
            kernel_size=int(kernel_size),
            resblock=str(resblock),
            resblock_kernel_sizes=[int(k) for k in resblock_kernel_sizes],
            resblock_dilation_sizes=[[int(d) for d in ds] for ds in resblock_dilation_sizes],
            upsample_rates=[int(u) for u in upsample_rates],
            upsample_initial_channel=int(upsample_initial_channel),
            upsample_kernel_sizes=[int(k) for k in upsample_kernel_sizes],
            spk_embed_dim=int(spk_embed_dim),
            gin_channels=int(gin_channels),
            sr=int(sr),
            version=version,
            f0=bool(f0),
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "SynthConfig":
        return cls(**json.loads(s))


def leaky_relu(x: mx.array, slope: float) -> mx.array:
    return mx.maximum(x, slope * x)


# --------------------------------------------------------------------------------------------------- text encoder


class LayerNorm(nn.Module):
    """Channel LayerNorm with RVC's parameter names (`gamma`, `beta`)."""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = mx.ones((channels,))
        self.beta = mx.zeros((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.layer_norm(x, self.gamma, self.beta, self.eps)


class MultiHeadAttention(nn.Module):
    """Self-attention with windowed relative position embeddings (VITS / RVC `attentions.MultiHeadAttention`)."""

    def __init__(self, channels: int, out_channels: int, n_heads: int, window_size: int = 10):
        super().__init__()
        self.n_heads = n_heads
        self.k_channels = channels // n_heads
        self.window_size = window_size
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        scale = self.k_channels**-0.5
        self.emb_rel_k = mx.random.normal((1, 2 * window_size + 1, self.k_channels)) * scale
        self.emb_rel_v = mx.random.normal((1, 2 * window_size + 1, self.k_channels)) * scale

    def _relative_embeddings(self, emb: mx.array, length: int) -> mx.array:
        w = self.window_size
        pad = max(length - (w + 1), 0)
        start = max((w + 1) - length, 0)
        if pad > 0:
            emb = mx.pad(emb, [(0, 0), (pad, pad), (0, 0)])
        return emb[:, start : start + 2 * length - 1]  # (1, 2L-1, k)

    @staticmethod
    def _relative_to_absolute(x: mx.array) -> mx.array:
        # (B, H, L, 2L-1) -> (B, H, L, L): out[i, j] = x[i, j - i + L - 1]
        B, H, L, _ = x.shape
        x = mx.pad(x, [(0, 0), (0, 0), (0, 0), (0, 1)]).reshape(B, H, 2 * L * L)
        x = mx.pad(x, [(0, 0), (0, 0), (0, L - 1)]).reshape(B, H, L + 1, 2 * L - 1)
        return x[:, :, :L, L - 1 :]

    @staticmethod
    def _absolute_to_relative(x: mx.array) -> mx.array:
        # (B, H, L, L) -> (B, H, L, 2L-1), the inverse re-indexing (zero where no absolute position exists).
        B, H, L, _ = x.shape
        x = mx.pad(x, [(0, 0), (0, 0), (0, 0), (0, L - 1)]).reshape(B, H, L * L + L * (L - 1))
        x = mx.pad(x, [(0, 0), (0, 0), (L, 0)]).reshape(B, H, L, 2 * L)
        return x[:, :, :, 1:]

    def __call__(self, x: mx.array, attn_mask: Optional[mx.array] = None) -> mx.array:
        B, T, C = x.shape
        H, K = self.n_heads, self.k_channels

        def heads(t):
            return t.reshape(B, T, H, K).transpose(0, 2, 1, 3)  # (B, H, T, K)

        q = heads(self.conv_q(x)) / math.sqrt(K)
        k = heads(self.conv_k(x))
        v = heads(self.conv_v(x))

        scores = q @ k.transpose(0, 1, 3, 2)
        rel_k = self._relative_embeddings(self.emb_rel_k, T)
        scores = scores + self._relative_to_absolute(q @ rel_k.transpose(0, 2, 1)[None])
        if attn_mask is not None:
            scores = mx.where(attn_mask == 0, mx.array(-1e4, scores.dtype), scores)
        p = mx.softmax(scores.astype(mx.float32), axis=-1).astype(v.dtype)
        out = p @ v
        rel_v = self._relative_embeddings(self.emb_rel_v, T)
        out = out + self._absolute_to_relative(p) @ rel_v[None]
        return self.conv_o(out.transpose(0, 2, 1, 3).reshape(B, T, C))


class FFN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, filter_channels: int, kernel_size: int):
        super().__init__()
        self.pad = ((kernel_size - 1) // 2, kernel_size // 2)
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)

    def _same(self, x: mx.array) -> mx.array:
        if self.pad == (0, 0):
            return x
        return mx.pad(x, [(0, 0), self.pad, (0, 0)])

    def __call__(self, x: mx.array, x_mask: mx.array) -> mx.array:
        x = nn.relu(self.conv_1(self._same(x * x_mask)))
        x = self.conv_2(self._same(x * x_mask))
        return x * x_mask


class Encoder(nn.Module):
    def __init__(self, hidden: int, filter_channels: int, n_heads: int, n_layers: int, kernel_size: int):
        super().__init__()
        self.attn_layers = [MultiHeadAttention(hidden, hidden, n_heads) for _ in range(n_layers)]
        self.norm_layers_1 = [LayerNorm(hidden) for _ in range(n_layers)]
        self.ffn_layers = [FFN(hidden, hidden, filter_channels, kernel_size) for _ in range(n_layers)]
        self.norm_layers_2 = [LayerNorm(hidden) for _ in range(n_layers)]

    def __call__(self, x: mx.array, x_mask: mx.array) -> mx.array:
        # (B, T, 1) mask -> (B, 1, T, T) pairwise attention mask.
        m = x_mask[:, None, :, 0]
        attn_mask = m[..., :, None] * m[..., None, :]
        x = x * x_mask
        for attn, n1, ffn, n2 in zip(self.attn_layers, self.norm_layers_1, self.ffn_layers, self.norm_layers_2):
            x = n1(x + attn(x, attn_mask))
            x = n2(x + ffn(x, x_mask))
        return x * x_mask


class TextEncoder(nn.Module):
    """`TextEncoder256` / `TextEncoder768`: content (+ coarse pitch) -> prior mean and log-scale."""

    def __init__(self, cfg: SynthConfig):
        super().__init__()
        self.hidden_channels = cfg.hidden_channels
        self.out_channels = cfg.inter_channels
        self.emb_phone = nn.Linear(cfg.feature_dim, cfg.hidden_channels)
        if cfg.f0:
            self.emb_pitch = nn.Embedding(256, cfg.hidden_channels)
        self.encoder = Encoder(cfg.hidden_channels, cfg.filter_channels, cfg.n_heads, cfg.n_layers, cfg.kernel_size)
        self.proj = nn.Conv1d(cfg.hidden_channels, cfg.inter_channels * 2, 1)

    def __call__(self, phone: mx.array, pitch: Optional[mx.array], lengths: mx.array):
        x = self.emb_phone(phone)
        if pitch is not None:
            x = x + self.emb_pitch(pitch)
        x = leaky_relu(x * math.sqrt(self.hidden_channels), 0.1)
        T = x.shape[1]
        x_mask = (mx.arange(T)[None, :] < lengths[:, None]).astype(x.dtype)[..., None]
        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = mx.split(stats, 2, axis=-1)
        return m, logs, x_mask


# ----------------------------------------------------------------------------------------------------------- flow


class WN(nn.Module):
    """WaveNet-style gated dilated conv stack (non-causal), conditioned on the speaker embedding."""

    def __init__(self, hidden: int, kernel_size: int, dilation_rate: int, n_layers: int, gin_channels: int):
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        if gin_channels:
            self.cond_layer = nn.Conv1d(gin_channels, 2 * hidden * n_layers, 1)
        self.in_layers = []
        self.res_skip_layers = []
        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = (kernel_size * dilation - dilation) // 2
            self.in_layers.append(nn.Conv1d(hidden, 2 * hidden, kernel_size, dilation=dilation, padding=padding))
            res_skip = 2 * hidden if i < n_layers - 1 else hidden
            self.res_skip_layers.append(nn.Conv1d(hidden, res_skip, 1))

    def __call__(self, x: mx.array, x_mask: mx.array, g: Optional[mx.array] = None) -> mx.array:
        output = mx.zeros_like(x)
        if g is not None:
            g = self.cond_layer(g)
        h = self.hidden
        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                x_in = x_in + g[..., i * 2 * h : (i + 1) * 2 * h]
            acts = mx.tanh(x_in[..., :h]) * mx.sigmoid(x_in[..., h:])
            res_skip = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                x = (x + res_skip[..., :h]) * x_mask
                output = output + res_skip[..., h:]
            else:
                output = output + res_skip
        return output * x_mask


class Flip(nn.Module):
    def __call__(self, x: mx.array, *args, **kwargs) -> mx.array:
        return x[..., ::-1]


class ResidualCouplingLayer(nn.Module):
    """Mean-only affine coupling layer, inverse direction (the only one inference needs)."""

    def __init__(self, channels: int, hidden: int, kernel_size: int, dilation_rate: int, n_layers: int, gin: int):
        super().__init__()
        self.half = channels // 2
        self.pre = nn.Conv1d(self.half, hidden, 1)
        self.enc = WN(hidden, kernel_size, dilation_rate, n_layers, gin)
        self.post = nn.Conv1d(hidden, self.half, 1)

    def __call__(self, x: mx.array, x_mask: mx.array, g: Optional[mx.array] = None) -> mx.array:
        x0, x1 = x[..., : self.half], x[..., self.half :]
        h = self.enc(self.pre(x0) * x_mask, x_mask, g=g)
        m = self.post(h) * x_mask
        return mx.concatenate([x0, (x1 - m) * x_mask], axis=-1)


class ResidualCouplingBlock(nn.Module):
    def __init__(self, channels: int, hidden: int, kernel_size: int, dilation_rate: int, n_layers: int, gin: int, n_flows: int = 4):
        super().__init__()
        # Coupling layers sit at even indices and parameter-free Flips at odd ones, so names match `flow.flows.{0,2,4,6}`.
        self.flows = []
        for _ in range(n_flows):
            self.flows.append(ResidualCouplingLayer(channels, hidden, kernel_size, dilation_rate, n_layers, gin))
            self.flows.append(Flip())

    def __call__(self, x: mx.array, x_mask: mx.array, g: Optional[mx.array] = None) -> mx.array:
        for flow in reversed(self.flows):
            x = flow(x, x_mask, g=g)
        return x


# -------------------------------------------------------------------------------------------------------- vocoder


class ResBlock1(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation=(1, 3, 5)):
        super().__init__()
        pad = lambda d: (kernel_size * d - d) // 2  # noqa: E731
        self.convs1 = [nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=pad(d)) for d in dilation]
        self.convs2 = [nn.Conv1d(channels, channels, kernel_size, padding=pad(1)) for _ in dilation]

    def __call__(self, x: mx.array) -> mx.array:
        for c1, c2 in zip(self.convs1, self.convs2):
            x = x + c2(leaky_relu(c1(leaky_relu(x, LRELU_SLOPE)), LRELU_SLOPE))
        return x


class ResBlock2(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation=(1, 3)):
        super().__init__()
        pad = lambda d: (kernel_size * d - d) // 2  # noqa: E731
        self.convs = [nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=pad(d)) for d in dilation]

    def __call__(self, x: mx.array) -> mx.array:
        for c in self.convs:
            x = x + c(leaky_relu(x, LRELU_SLOPE))
        return x


def sine_excitation(
    f0: mx.array,
    upp: int,
    sr: int,
    sine_amp: float = 0.1,
    noise_std: float = 0.003,
    noise: Optional[mx.array] = None,
    noise_scale: float = 1.0,
) -> mx.array:
    """
    NSF harmonic source (`SineGen`, harmonic_num=0): a phase-continuous sine at f0, upsampled by `upp`, with Gaussian
    noise added (small in voiced frames, the whole signal in unvoiced ones).

    Phase is accumulated per frame and wrapped to [-0.5, 0.5) before the cumulative sum, which keeps float32 precise on
    long inputs. `f0` is `(B, T)` Hz; returns `(B, T * upp, 1)`.
    """
    B, T = f0.shape
    f0 = f0.astype(mx.float32)[..., None]  # (B, T, 1)
    rad = f0 / sr * mx.arange(1, upp + 1, dtype=mx.float32)  # (B, T, upp): phase (in turns) within each frame
    frame_end = rad[:, :-1, -1:] + 0.5
    frame_end = frame_end - mx.floor(frame_end) - 0.5  # fmod(x + 0.5, 1) - 0.5 for x >= 0
    acc = mx.cumsum(frame_end, axis=1)
    acc = acc - _trunc(acc)  # torch.fmod(acc, 1.0)
    rad = rad + mx.pad(acc, [(0, 0), (1, 0), (0, 0)])
    sines = mx.sin(2 * math.pi * rad.reshape(B, T * upp, 1)) * sine_amp
    uv = mx.repeat((f0 > 0).astype(mx.float32), upp, axis=1)  # (B, T*upp, 1), nearest upsampling
    if noise is None:
        noise = mx.random.normal(sines.shape) if noise_scale else mx.zeros(sines.shape)
    noise = noise * noise_scale * (uv * noise_std + (1 - uv) * sine_amp / 3)
    return sines * uv + noise


def _trunc(x: mx.array) -> mx.array:
    return mx.where(x >= 0, mx.floor(x), mx.ceil(x))


class SourceModuleHnNSF(nn.Module):
    def __init__(self, sr: int):
        super().__init__()
        self.sr = sr
        self.l_linear = nn.Linear(1, 1)

    def __call__(self, f0: mx.array, upp: int, noise=None, noise_scale: float = 1.0) -> mx.array:
        sine = sine_excitation(f0, upp, self.sr, noise=noise, noise_scale=noise_scale)
        return mx.tanh(self.l_linear(sine.astype(self.l_linear.weight.dtype)))


class Generator(nn.Module):
    """HiFi-GAN generator; with `nsf=True` it is RVC's `GeneratorNSF` (sine-excitation injected at every scale)."""

    def __init__(self, cfg: SynthConfig, nsf: bool):
        super().__init__()
        self.nsf = nsf
        self.num_kernels = len(cfg.resblock_kernel_sizes)
        self.upp = cfg.hop_length
        c0 = cfg.upsample_initial_channel
        self.conv_pre = nn.Conv1d(cfg.inter_channels, c0, 7, padding=3)
        if cfg.gin_channels:
            self.cond = nn.Conv1d(cfg.gin_channels, c0, 1)
        self.ups = []
        self.resblocks = []
        if nsf:
            self.m_source = SourceModuleHnNSF(cfg.sr)
            self.noise_convs = []
        block = ResBlock1 if cfg.resblock == "1" else ResBlock2
        ch = c0
        for i, (u, k) in enumerate(zip(cfg.upsample_rates, cfg.upsample_kernel_sizes)):
            ch = c0 // (2 ** (i + 1))
            self.ups.append(nn.ConvTranspose1d(c0 // (2**i), ch, k, stride=u, padding=(k - u) // 2))
            if nsf:
                if i + 1 < len(cfg.upsample_rates):
                    s = math.prod(cfg.upsample_rates[i + 1 :])
                    self.noise_convs.append(nn.Conv1d(1, ch, s * 2, stride=s, padding=s // 2))
                else:
                    self.noise_convs.append(nn.Conv1d(1, ch, 1))
            for kk, d in zip(cfg.resblock_kernel_sizes, cfg.resblock_dilation_sizes):
                self.resblocks.append(block(ch, kk, d))
        self.conv_post = nn.Conv1d(ch, 1, 7, padding=3, bias=False)

    def __call__(self, x: mx.array, f0: Optional[mx.array] = None, g: Optional[mx.array] = None, noise=None, noise_scale: float = 1.0) -> mx.array:
        if self.nsf:
            har = self.m_source(f0, self.upp, noise=noise, noise_scale=noise_scale)  # (B, T*upp, 1)
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)
        for i, up in enumerate(self.ups):
            x = up(leaky_relu(x, LRELU_SLOPE))
            if self.nsf:
                x = x + self.noise_convs[i](har)
            blocks = self.resblocks[i * self.num_kernels : (i + 1) * self.num_kernels]
            xs = blocks[0](x)
            for b in blocks[1:]:
                xs = xs + b(x)
            x = xs / self.num_kernels
        x = self.conv_post(leaky_relu(x, 0.01))
        return mx.tanh(x)


# ---------------------------------------------------------------------------------------------------- synthesizer


class Synthesizer(nn.Module):
    def __init__(self, cfg: SynthConfig):
        super().__init__()
        self.cfg = cfg
        self.enc_p = TextEncoder(cfg)
        self.dec = Generator(cfg, nsf=cfg.f0)
        self.flow = ResidualCouplingBlock(cfg.inter_channels, cfg.hidden_channels, 5, 1, 3, cfg.gin_channels)
        self.emb_g = nn.Embedding(cfg.spk_embed_dim, cfg.gin_channels)

    def infer(
        self,
        phone: mx.array,
        phone_lengths: mx.array,
        pitch: Optional[mx.array] = None,
        nsff0: Optional[mx.array] = None,
        sid: Optional[mx.array] = None,
        noise_scale: float = 0.66666,
        prior_noise: Optional[mx.array] = None,
        nsf_noise: Optional[mx.array] = None,
        nsf_noise_scale: float = 1.0,
        speaker_embedding: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Synthesize audio.

        :param phone: `(B, T, feature_dim)` content features at 100 frames/s.
        :param phone_lengths: `(B,)` valid frame counts.
        :param pitch: `(B, T)` coarse pitch bins in [1, 255] (f0 voices only).
        :param nsff0: `(B, T)` f0 in Hz (f0 voices only).
        :param sid: `(B,)` speaker ids. Ignored when `speaker_embedding` is given.
        :param noise_scale: prior temperature (RVC uses 0.66666). `prior_noise` overrides the random draw.
        :param nsf_noise: explicit excitation noise `(B, T * hop, 1)`; `nsf_noise_scale=0` disables it.
        :param speaker_embedding: `(B, gin_channels)` embedding to use instead of looking up `sid` (voice blending).
        :returns: `(B, T * hop_length)` waveform in [-1, 1].
        """
        if speaker_embedding is None:
            if sid is None:
                sid = mx.zeros((phone.shape[0],), dtype=mx.int32)
            speaker_embedding = self.emb_g(sid)
        g = speaker_embedding[:, None, :]
        m_p, logs_p, x_mask = self.enc_p(phone, pitch if self.cfg.f0 else None, phone_lengths)
        if prior_noise is None:
            prior_noise = mx.random.normal(m_p.shape).astype(m_p.dtype) if noise_scale else mx.zeros_like(m_p)
        z_p = (m_p + mx.exp(logs_p) * prior_noise * noise_scale) * x_mask
        z = self.flow(z_p, x_mask, g=g)
        if self.cfg.f0:
            o = self.dec(z * x_mask, nsff0, g=g, noise=nsf_noise, noise_scale=nsf_noise_scale)
        else:
            o = self.dec(z * x_mask, g=g)
        return o[..., 0]
