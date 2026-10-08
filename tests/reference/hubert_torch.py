"""
Minimal PyTorch re-implementation of fairseq's `HubertModel` inference path, as used by RVC (`hubert_base.pt`).

Module nesting and attribute names follow fairseq exactly so that `state_dict()` keys are identical to the released
checkpoint (`feature_extractor.conv_layers.0.0.weight`, `encoder.pos_conv.0.weight_g`, ...). Only the pieces RVC
touches at inference are kept: no masking, no label embeddings, no dropout.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tests.reference import legacy_weight_norm


class Fp32GroupNorm(nn.GroupNorm):
    def forward(self, input):
        output = F.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


class ConvFeatureExtractionModel(nn.Module):
    def __init__(self, conv_layers):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        in_d = 1
        for i, (dim, k, stride) in enumerate(conv_layers):
            conv = nn.Conv1d(in_d, dim, k, stride=stride, bias=False)
            nn.init.kaiming_normal_(conv.weight)
            if i == 0:
                block = nn.Sequential(conv, nn.Dropout(0.0), Fp32GroupNorm(dim, dim, affine=True), nn.GELU())
            else:
                block = nn.Sequential(conv, nn.Dropout(0.0), nn.GELU())
            self.conv_layers.append(block)
            in_d = dim

    def forward(self, x):
        x = x.unsqueeze(1)
        for conv in self.conv_layers:
            x = conv(x)
        return x


class SamePad(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x: (T, B, C) as in fairseq.
        tgt_len, bsz, _ = x.shape
        q = self.q_proj(x) * self.scaling
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)), dim=-1)
        out = torch.bmm(attn, v).transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
        return self.out_proj(out)


class TransformerSentenceEncoderLayer(nn.Module):
    def __init__(self, embed_dim, ffn_dim, num_heads):
        super().__init__()
        self.self_attn = MultiheadAttention(embed_dim, num_heads)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        residual = x
        x = self.self_attn(x)
        x = self.self_attn_layer_norm(residual + x)
        residual = x
        x = self.fc2(F.gelu(self.fc1(x)))
        x = self.final_layer_norm(residual + x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, ffn_dim, num_heads, num_layers, conv_pos, conv_pos_groups):
        super().__init__()
        pos_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=conv_pos, padding=conv_pos // 2, groups=conv_pos_groups)
        std = math.sqrt(4 / (conv_pos * embed_dim))
        nn.init.normal_(pos_conv.weight, mean=0, std=std)
        nn.init.normal_(pos_conv.bias, std=0.1)
        pos_conv = legacy_weight_norm(pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(pos_conv, SamePad(conv_pos), nn.GELU())
        self.layers = nn.ModuleList(
            [TransformerSentenceEncoderLayer(embed_dim, ffn_dim, num_heads) for _ in range(num_layers)]
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, tgt_layer=None):
        x_conv = self.pos_conv(x.transpose(1, 2)).transpose(1, 2)
        x = x + x_conv
        x = self.layer_norm(x)
        x = x.transpose(0, 1)
        r = None
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == tgt_layer:
                r = x
                break
        if r is not None:
            x = r
        return x.transpose(0, 1)


class TorchHubert(nn.Module):
    def __init__(
        self,
        conv_layers=((512, 10, 5),) + ((512, 3, 2),) * 4 + ((512, 2, 2),) * 2,
        embed_dim=768,
        ffn_dim=3072,
        num_heads=12,
        num_layers=12,
        conv_pos=128,
        conv_pos_groups=16,
        final_dim=256,
    ):
        super().__init__()
        self.feature_extractor = ConvFeatureExtractionModel(conv_layers)
        self.layer_norm = nn.LayerNorm(conv_layers[-1][0])
        self.post_extract_proj = nn.Linear(conv_layers[-1][0], embed_dim)
        self.encoder = TransformerEncoder(embed_dim, ffn_dim, num_heads, num_layers, conv_pos, conv_pos_groups)
        self.final_proj = nn.Linear(embed_dim, final_dim)

    def extract_features(self, source, padding_mask=None, output_layer=None):
        features = self.feature_extractor(source).transpose(1, 2)
        features = self.layer_norm(features)
        features = self.post_extract_proj(features)
        x = self.encoder(features, tgt_layer=None if output_layer is None else output_layer - 1)
        return x, None
