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
