"""
Backwards-compatible re-export.

The bridge helpers moved to `rvc_mlx._convert_bridge` so the checkpoint converter can reuse them. Tests still import
from this module; it just forwards everything.
"""

from rvc_mlx._convert_bridge import *  # noqa: F401,F403
from rvc_mlx._convert_bridge import (  # noqa: F401 — explicit re-export for IDE/lint discovery
    copy_conv1d,
    copy_conv2d,
    copy_conv_transpose2d,
    copy_batchnorm,
    randomize_bn_stats,
    copy_conv_block_res,
    copy_res_encoder_block,
    copy_res_decoder_block,
    copy_encoder,
    copy_intermediate,
    copy_decoder,
    copy_deep_unet,
    copy_bi_gru,
    copy_linear,
    copy_e2e,
    copy_layer_norm,
    copy_ffn,
    copy_multi_head_attention,
    copy_transformer_encoder,
    copy_embedding,
    copy_text_encoder_768,
    copy_source_module_hn_nsf,
    copy_conv_transpose1d,
    copy_res_block1,
    copy_generator_nsf,
    copy_wn,
    copy_residual_coupling_layer,
    copy_residual_coupling_block,
    copy_synthesizer_trn_ms768_nsfsid,
    copy_group_norm,
    copy_layer_norm_pytorch_style,
    copy_feature_extractor,
    copy_hubert_multi_head_attention,
    copy_transformer_sentence_encoder_layer,
    copy_positional_conv,
    copy_hubert_transformer_encoder,
    copy_hubert_model,
    to_channels_last,
    to_channels_first,
    to_time_last,
    to_time_first,
    set_eval,
)
