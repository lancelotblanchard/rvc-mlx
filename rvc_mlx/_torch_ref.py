"""
PyTorch reference implementations of the RVC RMVPE modules.

Two consumers — the test suite (paired-module bridging) and the checkpoint converter (`rvc_mlx.convert`). Both need
torch installed; the runtime inference path does not import this module.

These classes mirror the original RVC reference (module layout, attribute names, `nn.Sequential` ordering) so that
state_dict keys from released `.pt` files load directly via `load_state_dict`. Any change to attribute names or
nesting here will silently break checkpoint compatibility.
"""

import torch


class TorchConvBlockRes(torch.nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.01):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ),
            torch.nn.BatchNorm2d(out_channels, momentum=momentum),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ),
            torch.nn.BatchNorm2d(out_channels, momentum=momentum),
            torch.nn.ReLU(),
        )
        if in_channels != out_channels:
            self.shortcut = torch.nn.Conv2d(in_channels, out_channels, (1, 1))

    def forward(self, x):
        if not hasattr(self, "shortcut"):
            return self.conv(x) + x
        return self.conv(x) + self.shortcut(x)


class TorchResEncoderBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_blocks=1, momentum=0.01):
        super().__init__()
        self.n_blocks = n_blocks
        self.conv = torch.nn.ModuleList()
        self.conv.append(TorchConvBlockRes(in_channels, out_channels, momentum))
        for _ in range(n_blocks - 1):
            self.conv.append(TorchConvBlockRes(out_channels, out_channels, momentum))
        self.kernel_size = kernel_size
        if self.kernel_size is not None:
            self.pool = torch.nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        if self.kernel_size is not None:
            return x, self.pool(x)
        return x


class TorchEncoder(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        in_size,
        n_encoders,
        kernel_size,
        n_blocks,
        out_channels=16,
        momentum=0.01,
    ):
        super().__init__()
        self.n_encoders = n_encoders
        self.bn = torch.nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = torch.nn.ModuleList()
        for _ in range(n_encoders):
            self.layers.append(
                TorchResEncoderBlock(in_channels, out_channels, kernel_size, n_blocks, momentum)
            )
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.out_size = in_size
        self.out_channel = out_channels

    def forward(self, x):
        concat_tensors = []
        x = self.bn(x)
        for layer in self.layers:
            t, x = layer(x)
            concat_tensors.append(t)
        return x, concat_tensors


class TorchIntermediate(torch.nn.Module):
    def __init__(self, in_channels, out_channels, n_inters, n_blocks, momentum=0.01):
        super().__init__()
        self.n_inters = n_inters
        self.layers = torch.nn.ModuleList()
        self.layers.append(TorchResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum))
        for _ in range(n_inters - 1):
            self.layers.append(TorchResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TorchResDecoderBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride, n_blocks=1, momentum=0.01):
        super().__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.n_blocks = n_blocks
        self.conv1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
                output_padding=out_padding,
                bias=False,
            ),
            torch.nn.BatchNorm2d(out_channels, momentum=momentum),
            torch.nn.ReLU(),
        )
        self.conv2 = torch.nn.ModuleList()
        self.conv2.append(TorchConvBlockRes(out_channels * 2, out_channels, momentum))
        for _ in range(n_blocks - 1):
            self.conv2.append(TorchConvBlockRes(out_channels, out_channels, momentum))

    def forward(self, x, concat_tensor):
        x = self.conv1(x)
        x = torch.cat((x, concat_tensor), dim=1)
        for conv2 in self.conv2:
            x = conv2(x)
        return x


class TorchDecoder(torch.nn.Module):
    def __init__(self, in_channels, n_decoders, stride, n_blocks, momentum=0.01):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.n_decoders = n_decoders
        for _ in range(n_decoders):
            out_channels = in_channels // 2
            self.layers.append(
                TorchResDecoderBlock(in_channels, out_channels, stride, n_blocks, momentum)
            )
            in_channels = out_channels

    def forward(self, x, concat_tensors):
        for i, layer in enumerate(self.layers):
            x = layer(x, concat_tensors[-1 - i])
        return x


class TorchDeepUnet(torch.nn.Module):
    def __init__(
        self,
        kernel_size,
        n_blocks,
        en_de_layers=5,
        inter_layers=4,
        in_channels=1,
        en_out_channels=16,
    ):
        super().__init__()
        self.encoder = TorchEncoder(
            in_channels, 128, en_de_layers, kernel_size, n_blocks, en_out_channels
        )
        self.intermediate = TorchIntermediate(
            self.encoder.out_channel // 2,
            self.encoder.out_channel,
            inter_layers,
            n_blocks,
        )
        self.decoder = TorchDecoder(
            self.encoder.out_channel, en_de_layers, kernel_size, n_blocks
        )

    def forward(self, x):
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        x = self.decoder(x, concat_tensors)
        return x


class TorchBiGRU(torch.nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super().__init__()
        self.gru = torch.nn.GRU(
            input_features,
            hidden_features,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        return self.gru(x)[0]


class TorchE2E(torch.nn.Module):
    """PyTorch reference for the RVC E2E pitch network (n_gru > 0 path only).

    The released RVC `rmvpe.pt` checkpoint serializes against this exact module layout. Changing attribute names,
    Sequential element order, or nesting will break `load_state_dict` against published weights.
    """

    def __init__(
        self,
        n_blocks,
        n_gru,
        kernel_size,
        en_de_layers=5,
        inter_layers=4,
        in_channels=1,
        en_out_channels=16,
    ):
        super().__init__()
        self.unet = TorchDeepUnet(
            kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, en_out_channels
        )
        self.cnn = torch.nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        self.fc = torch.nn.Sequential(
            TorchBiGRU(3 * 128, 256, n_gru),
            torch.nn.Linear(512, 360),
            torch.nn.Dropout(0.25),
            torch.nn.Sigmoid(),
        )

    def forward(self, mel):
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        return self.fc(x)


# Default hyperparameters used by the released RVC `rmvpe.pt` checkpoint. Pinned here so the converter and the
# `RMVPE.from_pretrained` factory build the same architecture without each caller having to remember the numbers.
DEFAULT_E2E_CONFIG = dict(
    n_blocks=4,
    n_gru=1,
    kernel_size=(2, 2),
    en_de_layers=5,
    inter_layers=4,
    in_channels=1,
    en_out_channels=16,
)


# ----------------------------------------------------------------------------------------------------------------------
# Synthesizer reference modules (transformer encoder building blocks).
#
# These mirror the RVC reference's `LayerNorm`, `FFN`, `MultiHeadAttention`, and transformer `Encoder` exactly so we can
# bridge weights via paired-module tests. RVC's MultiHeadAttention uses the **optional relative-position embeddings**
# from Shaw et al. 2018, switched on when `window_size` is not None.
# ----------------------------------------------------------------------------------------------------------------------


class TorchLayerNorm(torch.nn.Module):
    """Per-channel LayerNorm. Operates on `(B, C, T)` input by transposing channel to last, normalizing, transposing
    back. Parameters are named `gamma`/`beta` to match the RVC reference."""

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = torch.nn.Parameter(torch.ones(channels))
        self.beta = torch.nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = torch.nn.functional.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class TorchFFN(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        filter_channels,
        kernel_size,
        p_dropout=0.0,
        activation=None,
        causal=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation
        self.causal = causal
        self.is_activation = activation == "gelu"

        self.conv_1 = torch.nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = torch.nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = torch.nn.Dropout(p_dropout)

    def _causal_padding(self, x):
        if self.kernel_size == 1:
            return x
        return torch.nn.functional.pad(x, [self.kernel_size - 1, 0, 0, 0, 0, 0])

    def _same_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size // 2
        return torch.nn.functional.pad(x, [pad_l, pad_r, 0, 0, 0, 0])

    def padding(self, x, x_mask):
        if self.causal:
            return self._causal_padding(x * x_mask)
        return self._same_padding(x * x_mask)

    def forward(self, x, x_mask):
        x = self.conv_1(self.padding(x, x_mask))
        if self.is_activation:
            x = x * torch.sigmoid(1.702 * x)
        else:
            x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(self.padding(x, x_mask))
        return x * x_mask


class TorchMultiHeadAttention(torch.nn.Module):
    def __init__(
        self,
        channels,
        out_channels,
        n_heads,
        p_dropout=0.0,
        window_size=None,
        heads_share=True,
        block_length=None,
        proximal_bias=False,
        proximal_init=False,
    ):
        super().__init__()
        assert channels % n_heads == 0
        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.heads_share = heads_share
        self.block_length = block_length
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init

        self.k_channels = channels // n_heads
        self.conv_q = torch.nn.Conv1d(channels, channels, 1)
        self.conv_k = torch.nn.Conv1d(channels, channels, 1)
        self.conv_v = torch.nn.Conv1d(channels, channels, 1)
        self.conv_o = torch.nn.Conv1d(channels, out_channels, 1)
        self.drop = torch.nn.Dropout(p_dropout)

        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels**-0.5
            self.emb_rel_k = torch.nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev
            )
            self.emb_rel_v = torch.nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev
            )

        torch.nn.init.xavier_uniform_(self.conv_q.weight)
        torch.nn.init.xavier_uniform_(self.conv_k.weight)
        torch.nn.init.xavier_uniform_(self.conv_v.weight)
        if proximal_init:
            with torch.no_grad():
                self.conv_k.weight.copy_(self.conv_q.weight)
                self.conv_k.bias.copy_(self.conv_q.bias)

    def forward(self, x, c, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)
        x = self.attention(q, k, v, mask=attn_mask)
        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):
        import math as _math

        b, d, t_s = key.size()
        t_t = query.size(2)
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        scores = torch.matmul(query / _math.sqrt(self.k_channels), key.transpose(-2, -1))
        if self.window_size is not None:
            assert t_s == t_t
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(
                query / _math.sqrt(self.k_channels), key_relative_embeddings
            )
            scores_local = self._relative_position_to_absolute_position(rel_logits)
            scores = scores + scores_local
        if self.proximal_bias:
            assert t_s == t_t
            scores = scores + self._attention_bias_proximal(t_s).to(
                device=scores.device, dtype=scores.dtype
            )
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
            if self.block_length is not None:
                assert t_s == t_t
                block_mask = (
                    torch.ones_like(scores)
                    .triu(-self.block_length)
                    .tril(self.block_length)
                )
                scores = scores.masked_fill(block_mask == 0, -1e4)
        p_attn = torch.nn.functional.softmax(scores, dim=-1)
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)
        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
            output = output + self._matmul_with_relative_values(
                relative_weights, value_relative_embeddings
            )
        output = output.transpose(2, 3).contiguous().view(b, d, t_t)
        return output

    def _matmul_with_relative_values(self, x, y):
        return torch.matmul(x, y.unsqueeze(0))

    def _matmul_with_relative_keys(self, x, y):
        return torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))

    def _get_relative_embeddings(self, relative_embeddings, length):
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start = max((self.window_size + 1) - length, 0)
        slice_end = slice_start + 2 * length - 1
        if pad_length > 0:
            padded = torch.nn.functional.pad(
                relative_embeddings, [0, 0, pad_length, pad_length, 0, 0]
            )
        else:
            padded = relative_embeddings
        return padded[:, slice_start:slice_end]

    def _relative_position_to_absolute_position(self, x):
        batch, heads, length, _ = x.size()
        x = torch.nn.functional.pad(x, [0, 1, 0, 0, 0, 0, 0, 0])
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = torch.nn.functional.pad(x_flat, [0, int(length) - 1, 0, 0, 0, 0])
        x_final = x_flat.view([batch, heads, length + 1, 2 * length - 1])[
            :, :, :length, length - 1 :
        ]
        return x_final

    def _absolute_position_to_relative_position(self, x):
        batch, heads, length, _ = x.size()
        x = torch.nn.functional.pad(
            x, [0, int(length) - 1, 0, 0, 0, 0, 0, 0]
        )
        x_flat = x.view([batch, heads, int(length**2) + int(length * (length - 1))])
        x_flat = torch.nn.functional.pad(x_flat, [length, 0, 0, 0, 0, 0])
        x_final = x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]
        return x_final

    def _attention_bias_proximal(self, length):
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)


def _torch_sequence_mask(length, max_length=None):
    """Local helper duplicating the RVC reference's `sequence_mask` so this module stays self-contained."""
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


class TorchTransformerEncoder(torch.nn.Module):
    """RVC's transformer encoder. Distinct from `TorchEncoder` (the RMVPE U-Net encoder)."""

    def __init__(
        self,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size=1,
        p_dropout=0.0,
        window_size=10,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = int(n_layers)
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size

        self.drop = torch.nn.Dropout(p_dropout)
        self.attn_layers = torch.nn.ModuleList()
        self.norm_layers_1 = torch.nn.ModuleList()
        self.ffn_layers = torch.nn.ModuleList()
        self.norm_layers_2 = torch.nn.ModuleList()
        for _ in range(self.n_layers):
            self.attn_layers.append(
                TorchMultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    window_size=window_size,
                )
            )
            self.norm_layers_1.append(TorchLayerNorm(hidden_channels))
            self.ffn_layers.append(
                TorchFFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(TorchLayerNorm(hidden_channels))

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for attn, norm1, ffn, norm2 in zip(
            self.attn_layers, self.norm_layers_1, self.ffn_layers, self.norm_layers_2
        ):
            y = attn(x, x, attn_mask)
            y = self.drop(y)
            x = norm1(x + y)

            y = ffn(x, x_mask)
            y = self.drop(y)
            x = norm2(x + y)
        x = x * x_mask
        return x


class TorchTextEncoder768(torch.nn.Module):
    """PyTorch reference for RVC's `TextEncoder768`. Mirrors the original source layout exactly."""

    def __init__(
        self,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
    ):
        super().__init__()
        import math as _math
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = float(p_dropout)
        self.emb_phone = torch.nn.Linear(768, hidden_channels)
        self.lrelu = torch.nn.LeakyReLU(0.1, inplace=True)
        self.emb_pitch = torch.nn.Embedding(256, hidden_channels)
        self.encoder = TorchTransformerEncoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            float(p_dropout),
        )
        self.proj = torch.nn.Conv1d(hidden_channels, out_channels * 2, 1)
        self._sqrt_hidden = _math.sqrt(hidden_channels)

    def forward(self, phone, pitch, lengths):
        if pitch is None:
            x = self.emb_phone(phone)
        else:
            x = self.emb_phone(phone) + self.emb_pitch(pitch)
        x = x * self._sqrt_hidden  # [B, T, hidden]
        x = self.lrelu(x)
        x = torch.transpose(x, 1, -1)  # [B, hidden, T]
        x_mask = torch.unsqueeze(_torch_sequence_mask(lengths, x.size(2)), 1).to(x.dtype)
        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return m, logs, x_mask


class TorchSineGen(torch.nn.Module):
    """
    PyTorch reference for `SineGen`. Modified to accept explicit `rand_ini` and `noise_raw` kwargs so paired-module
    tests can share the random tensors with the MLX impl (the original RVC implementation samples internally and is
    intrinsically non-deterministic).
    """

    def __init__(
        self,
        samp_rate,
        harmonic_num=0,
        sine_amp=0.1,
        noise_std=0.003,
        voiced_threshold=0,
        flag_for_pulse=False,
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse

    def _f02uv(self, f0):
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def forward(self, f0, upp, rand_ini=None, noise_raw=None):
        with torch.no_grad():
            f0 = f0[:, None].transpose(1, 2)  # (B, T, 1)
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device, dtype=f0.dtype)
            f0_buf[:, :, 0] = f0[:, :, 0]
            for idx in range(self.harmonic_num):
                f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (idx + 2)
            rad_values = (f0_buf / self.sampling_rate) % 1
            if rand_ini is None:
                rand_ini = torch.rand(f0_buf.shape[0], f0_buf.shape[2], device=f0_buf.device)
            rand_ini = rand_ini.clone()
            rand_ini[:, 0] = 0
            rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

            tmp_over_one = torch.cumsum(rad_values, 1)
            tmp_over_one *= upp
            tmp_over_one = torch.nn.functional.interpolate(
                tmp_over_one.transpose(2, 1),
                scale_factor=float(upp),
                mode="linear",
                align_corners=True,
            ).transpose(2, 1)
            rad_values = torch.nn.functional.interpolate(
                rad_values.transpose(2, 1), scale_factor=float(upp), mode="nearest"
            ).transpose(2, 1)
            tmp_over_one %= 1
            tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
            cumsum_shift = torch.zeros_like(rad_values)
            cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0

            sine_waves = torch.sin(
                torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * torch.pi
            )
            sine_waves = sine_waves * self.sine_amp

            uv = self._f02uv(f0)
            uv = torch.nn.functional.interpolate(
                uv.transpose(2, 1), scale_factor=float(upp), mode="nearest"
            ).transpose(2, 1)

            if noise_raw is None:
                noise_raw = torch.randn_like(sine_waves)
            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = noise_amp * noise_raw
            sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class TorchSourceModuleHnNSF(torch.nn.Module):
    """PyTorch reference for `SourceModuleHnNSF`. Threads `rand_ini` / `noise_raw` through to `TorchSineGen`."""

    def __init__(
        self,
        sampling_rate,
        harmonic_num=0,
        sine_amp=0.1,
        add_noise_std=0.003,
        voiced_threshold=0,
        is_half=False,
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.is_half = is_half
        self.l_sin_gen = TorchSineGen(
            sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshold
        )
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x, upp=1, rand_ini=None, noise_raw=None):
        sine_wavs, _, _ = self.l_sin_gen(x, upp, rand_ini=rand_ini, noise_raw=noise_raw)
        sine_wavs = sine_wavs.to(dtype=self.l_linear.weight.dtype)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        return sine_merge, None, None


def _torch_get_padding(kernel_size, dilation=1):
    return (kernel_size * dilation - dilation) // 2


class TorchResBlock1(torch.nn.Module):
    """PyTorch reference for ResBlock1. The original RVC source wraps each Conv1d in `weight_norm`; this reference uses
    plain Conv1d so the paired-module test compares fused (post-`remove_weight_norm`) weights."""

    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = tuple(dilation)
        self.lrelu_slope = 0.1
        self.convs1 = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=d,
                    padding=_torch_get_padding(kernel_size, d),
                )
                for d in self.dilation
            ]
        )
        self.convs2 = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=_torch_get_padding(kernel_size, 1),
                )
                for _ in self.dilation
            ]
        )

    def forward(self, x, x_mask=None):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = torch.nn.functional.leaky_relu(x, self.lrelu_slope)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = torch.nn.functional.leaky_relu(xt, self.lrelu_slope)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x


class TorchGeneratorNSF(torch.nn.Module):
    """PyTorch reference for GeneratorNSF (fused-weight variant; no weight_norm)."""

    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels,
        sr,
        is_half=False,
    ):
        super().__init__()
        import math as _math
        assert resblock == "1", "Only ResBlock1 is supported."
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.upsample_rates = tuple(upsample_rates)
        self.upp = _math.prod(upsample_rates)
        self.gin_channels = gin_channels
        self.lrelu_slope = 0.1

        self.m_source = TorchSourceModuleHnNSF(sampling_rate=sr, harmonic_num=0, is_half=is_half)
        self.conv_pre = torch.nn.Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )

        self.ups = torch.nn.ModuleList()
        self.noise_convs = torch.nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            in_ch = upsample_initial_channel // (2**i)
            out_ch = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(
                torch.nn.ConvTranspose1d(in_ch, out_ch, k, u, padding=(k - u) // 2)
            )
            if i + 1 < len(upsample_rates):
                stride_f0 = _math.prod(upsample_rates[i + 1 :])
                self.noise_convs.append(
                    torch.nn.Conv1d(
                        1,
                        out_ch,
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=stride_f0 // 2,
                    )
                )
            else:
                self.noise_convs.append(torch.nn.Conv1d(1, out_ch, kernel_size=1))

        self.resblocks = torch.nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(TorchResBlock1(ch, k, tuple(d)))

        final_ch = upsample_initial_channel // (2 ** len(upsample_rates))
        self.conv_post = torch.nn.Conv1d(final_ch, 1, 7, 1, padding=3, bias=False)

        if gin_channels != 0:
            self.cond = torch.nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, f0, g=None, rand_ini=None, noise_raw=None):
        har_source, _, _ = self.m_source(f0, self.upp, rand_ini=rand_ini, noise_raw=noise_raw)
        har_source = har_source.transpose(1, 2)  # (B, T*upp, 1) -> (B, 1, T*upp) channels-first
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = torch.nn.functional.leaky_relu(x, self.lrelu_slope)
            x = self.ups[i](x)
            x_source = self.noise_convs[i](har_source)
            x = x + x_source
            block_start = i * self.num_kernels
            xs = None
            for j in range(self.num_kernels):
                rb_out = self.resblocks[block_start + j](x)
                xs = rb_out if xs is None else xs + rb_out
            x = xs / self.num_kernels

        x = torch.nn.functional.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x
