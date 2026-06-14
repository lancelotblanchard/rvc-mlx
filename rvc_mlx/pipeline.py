import logging

from rvc_mlx.rmvpe import RMVPE

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)


class Pipeline(object):
    def __init__(self, tgt_sr, config):
        self.x_pad, self.x_query, self.x_center, self.x_max, self.is_half = (
            config.x_pad,
            config.x_query,
            config.x_center,
            config.x_max,
            config.is_half,
        )
        self.sr = 16_000
        self.window = 16
        self.t_pad = self.sr * self.x_pad
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * self.x_query
        self.t_center = self.sr * self.x_center
        self.t_max = self.sr * self.x_max

        logger.info(f"Loading RMVPE model {config.rmvpe_root}/rmvpe.pt")
        # `from_pretrained` accepts either a `.safetensors` file or the original RVC `.pt`. For `.pt` it auto-converts
        # to a sibling `.safetensors` on first use (requires torch installed); subsequent runs load directly.
        self.model_rmvpe = RMVPE.from_pretrained(
            f"{config.rmvpe_root}/rmvpe.pt",
            is_half=self.is_half,
        )

    def get_f0(self, x, f0_up_key, inp_f0=None):
        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)

        f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)

        f0 *= pow(2, f0_up_key / 12)
        tf0 = self.sr // self.window  # f0 per second
        if inp_f0 is not None:
            delta_t = np.round((inp_f0[:, 0].max() - inp_f0[:, 0].min()) * tf0 + 1).astype("int16")
            replace_f0 = np.interp(list(range(delta_t)), inp_f0[:, 0] * 100, inp_f0[:, 1])
            shape = f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)].shape[0]
            f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)] = replace_f0[:shape]

        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int32)
        return f0_coarse, f0bak
